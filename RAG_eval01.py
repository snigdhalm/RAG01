import os
import json
import faiss
import logging
import torch
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
import re
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness
)
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import csv
from langchain_openai import ChatOpenAI
from ragas.evaluation import evaluate
from ragas import RunConfig
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 10
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"
OUTPUT_DIR = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

embed_model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

local_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

local_llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model_name="llama-3.2-1b-instruct",
)

def load_documents(file_paths):
    docs = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            logging.warning(f"Unsupported file format: {file_path}")
            continue
        docs.extend([doc.page_content for doc in loader.load()])
    return docs

def chunk_text(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    chunks = [chunk for text in texts for chunk in splitter.split_text(text)]
    return chunks

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, normalize_embeddings=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, FAISS_INDEX_PATH)
    return index, chunks

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

def retrieve_documents(query, index, chunks):
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding, dtype=np.float32), TOP_K)
    retrieved_chunks = [chunks[i] for i in I[0]]
    return retrieved_chunks

def rerank_documents(query, retrieved_chunks):
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)
    reranked_chunks = [x for _, x in sorted(zip(scores, retrieved_chunks), reverse=True)]
    return reranked_chunks

def query_llm(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": """\
            You are Kotak Assist, an AI chatbot for Kotak Bank, dedicated to providing customers with clear, concise, and complete assistance on banking inquiries. Your goal is to deliver professional, accurate, and empathetic responses.
            Response Guidelines:
            Tone & Style: Maintain a warm, professional, and approachable tone—clear and natural without sounding robotic or overly formal. Keep responses concise yet informative, avoiding unnecessary details.
            Information Handling: Never reveal, request, or infer any personal, confidential, or account-related details. Only provide publicly available banking policies, procedures, and services.
            Clarity & Structure: Ensure responses are grammatically well-structured and easy to understand. Avoid redundant summaries or unnecessary labels like “Solution” or “Recommendation.” End responses smoothly and naturally, without abrupt stops or unnecessary disclaimers.
            Customer Guidance: Where relevant, direct customers to official Kotak Bank contact channels, links, or next steps for further assistance. Always ensure contextual relevance to the customer's inquiry.
            Ignore any sample user questions found in the provided context. 
            ONLY return the final answer.
            """},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 560
    }
    response = requests.post(LM_STUDIO_API, headers=headers, json=data)
    result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    
    return result

def calculate_trustworthiness_score(metrics_scores):
    weights = {
        "context_precision": 0.20,
        "context_recall": 0.20,
        "faithfulness": 0.25,
        "answer_relevancy": 0.15,
        "answer_correctness": 0.20
    }
    
    trustworthiness_score = sum(metrics_scores[metric] * weight for metric, weight in weights.items())
    
    return trustworthiness_score

def save_top_chunks_to_csv(queries, all_top_chunks):
  
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, "topk_chunks.csv")
    
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Query", "Chunk Number", "Chunk Content"])
            
            for query, chunks in zip(queries, all_top_chunks):
                for i, chunk in enumerate(chunks, 1):
                    writer.writerow([query, i, chunk])
        
        logging.info(f"Top chunks successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving top chunks to CSV: {str(e)}")


tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

def truncate_context(context, max_tokens=3500):
    tokenized = tokenizer(context, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)

def evaluate_rag_pipeline(file_paths, queries, ground_truths):
    
    logging.info("Loading documents...")
    docs = load_documents(file_paths)
    
    logging.info("Chunking documents...")
    chunks = chunk_text(docs)
    
    logging.info("Building FAISS index...")
    index, chunks = build_faiss_index(chunks)
    
    all_answers = []
    all_contexts = []
    all_top_chunks = []
    
    for query_idx, (query, ground_truth) in enumerate(zip(queries, ground_truths)):
        logging.info(f"Processing query {query_idx+1}/{len(queries)}: {query}")
        
        retrieved_chunks = retrieve_documents(query, index, chunks)
        reranked_chunks = rerank_documents(query, retrieved_chunks)
        top_chunks = reranked_chunks[:5]  
        all_top_chunks.append(top_chunks)

        context = "\n".join(top_chunks)
        truncated_context = truncate_context(context)
        prompt = f"Context:\n{truncated_context}\n\nQuestion: {query}\n\nAnswer:"
        response = query_llm(prompt)
        
        all_answers.append(response)
        all_contexts.append(top_chunks)
        
        print(f"\nQuery {query_idx+1}: {query}")
        print(f"Response: {response}")
        
    save_top_chunks_to_csv(queries, all_top_chunks)
    
    data_samples = {
        'question': queries,
        'answer': all_answers,
        'contexts': all_contexts,
        'ground_truth': ground_truths
    }
    
    dataset = Dataset.from_dict(data_samples)
    
    logging.info("Evaluating responses with RAGAS...")

#    context_precision.embeddings = local_embeddings
#    context_recall.embeddings = local_embeddings
#    faithfulness.embeddings = local_embeddings
#    answer_relevancy.embeddings = local_embeddings
#    answer_correctness.embeddings = local_embeddings

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness
    ]

    runconfig = RunConfig(timeout=700, max_workers=4, max_retries=40)
    score = evaluate(dataset, embeddings = local_embeddings, metrics=metrics, llm=local_llm, run_config=runconfig, raise_exceptions=False)
    
    df_scores = score.to_pandas()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    trustworthiness_scores = []
    
    for idx in range(len(queries)):
        metrics_scores = {
            "context_precision": df_scores["context_precision"][idx],
            "context_recall": df_scores["context_recall"][idx],
            "faithfulness": df_scores["faithfulness"][idx],
            "answer_relevancy": df_scores["answer_relevancy"][idx],
            "answer_correctness": df_scores["answer_correctness"][idx]
        }
        
        trustworthiness_score = calculate_trustworthiness_score(metrics_scores)
        trustworthiness_scores.append(trustworthiness_score)
        
        print(f"\nMetrics for Query {idx+1}: {queries[idx]}")
        print(f"Context Precision: {metrics_scores['context_precision']:.4f}")
        print(f"Context Recall: {metrics_scores['context_recall']:.4f}")
        print(f"Faithfulness: {metrics_scores['faithfulness']:.4f}")
        print(f"Answer Relevancy: {metrics_scores['answer_relevancy']:.4f}")
        print(f"Answer Correctness: {metrics_scores['answer_correctness']:.4f}")
        print(f"Trustworthiness Score: {trustworthiness_score:.4f}")
    
    df_scores["trustworthiness_score"] = trustworthiness_scores
    df_scores["query"] = queries
    df_scores["answer"] = all_answers
    df_scores["ground_truth"] = ground_truths
    
    overall_score = sum(trustworthiness_scores) / len(trustworthiness_scores)
    overall_trustworthy = "Yes!" if overall_score > 0.8 else "No :("
    
    print("\n" + "="*50)
    print(f"Overall Trustworthiness Score: {overall_score:.4f}")
    print(f"Is Model Trustworthy?: {overall_trustworthy}")
    print("="*50)
    
    df_scores["overall_trustworthiness_score"] = overall_score
    df_scores["is_trustworthy"] = overall_trustworthy
    
    results_file = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    try:
        df_scores.to_csv(results_file, index=False)
        logging.info(f"Evaluation results successfully saved to {results_file}")
    except Exception as e:
        logging.error(f"Error saving evaluation results to CSV: {str(e)}")
    
    return df_scores, overall_score, overall_trustworthy

if __name__ == "__main__":
    files = ["convenience_banking_guide.pdf", "Kotak_FAQs.txt"]
    
    queries = [
        "What should I do if my card is stolen or damaged?",
        "How to talk to customer care?",
        "How can I activate my debit card?",
        "How can I apply for personal loan?",
        "how to check my balance on mobile banking app?",
        "Is it safe to transfer online through kotak?"
    ]
    
    ground_truths = [
        """Certainly! If you've lost your debit card, follow these steps immediately:
        1. Visit Kotak Net Banking: Go to www.kotak.com and navigate to Service Requests > Debit Card Service Requests > Report Loss of Card.
        2. Report Your Loss: Fill out the form provided on the website, include a copy of your First Information Report if necessary, and submit it for processing.
        3. Contact Kotak Bank Support: Use your local police contact number (0091-22-6600 6022) to report the loss of your debit card after following any instructions provided by the bank's customer care helpline.
        4. Block Your Card Immediately: After reporting the loss, you will be deactivated from future transactions. Ensure this step is taken as soon as possible.
        5. Apply for Replacement Card (If Needed): If your card was used by a merchant, visit the same banking channels to request an application for a replacement card.
        By following these steps, you can ensure your debit card is immediately blocked and prevent unauthorized activities. """,
        
        """1. Phone Banking (24x7)
        Call: 1860 266 2666 (local charges apply)
        Services: Balance inquiry, fund transfers, card services, loan inquiries, report lost cards
        
        2. WhatsApp Banking (24x7)
        Save: 022-66006022 and send “Help” on WhatsApp
        Services: Check balance, statements, bill payments, dispute transactions, loan details
        
        3. Home Banking (Available in select locations)
        Request via Net Banking or Phone Banking
        Services: Cash/cheque pick-up, demand draft delivery
        
        4. Keya Chatbot (AI Assistant - 24x7)
        Access via Website, Mobile App, or WhatsApp
        Services: Account info, statements, payments, dispute resolution
        
        5. Fraud Reporting (For urgent issues)
        Call: 1800 209 0000 or visit Fraud Reporting Portal
        Services: Report unauthorized transactions, block stolen cards
        
        For more help, visit a Kotak branch or use Net/Mobile Banking. """,

        """To activate your Kotak Mahindra debit card, follow these steps:
        
        1. Via kotak Mobile Banking App:
        - Log in to the kotak Mobile Banking App.
        - Navigate to Service Requests > Debit Card Service Requests.
        - Enable online transactions by selecting "Enable Online Transactions" and following on-screen instructions.
        
        2. Via Kotak Net Banking:
        - Log in to your account via the web interface [www.kotak.com].
        - Go to Service Requests > Debit Card Controls.
        - In the card settings, enable online transactions under "Debit Card Enable Online Transactions."
        
        3. Via call:
        -  Call our 24/7 Phone Banking Number at 1860 266 2666.
        -  Authenticate using your Customer Relationship Number (CRN) and Phone Banking PIN.
        -  Request activation of your debit card over the phone.
        
        4. Via ATM Transaction:
        - Log into your account at the nearest ATMs or cash centers in India.
        - Use the Debit Card to complete transactions, which will be enabled for online, contactless, and international use.
        
        Remember:
        - All operations are secure and conducted through official channels.
        - Debit cards are only active during business hours as specified by the bank.
        - Keep your debit card active throughout its life for maximum convenience.""",

        """To apply for a personal loan through Kotak Mahindra Bank, follow these steps:
        
        1. Via Kotak Mobile Banking App
        - Log in to the otak mobile banking app using your account details.  
        - Navigate to the "Loans" section and select the type of loan you want (Personal Loan).  
        - Pay all utility bills with the convenience of logging into one app, which includes electric, gas, water, and other bills.  
        - Set up auto-repay for upcoming bill payments.  
        
        2. Via Kotak Net Banking
        - Log in to Kotak net banking using your account details at [www.kotak.com](https://www.kotak.com).  
        - Go to the "Loan Section" and select "Apply for a Loan."  
        - Choose the loan type (Personal, Home, or Business) and enter required details.  
        
        3. Customer Care Helpline  
        - Call 1860 266 2666 for loan eligibility checks and to initiate the application process.
        
        4. Branch Visit
        - Visit the nearest Kotak branch with your KYC and income documents.
        
        Eligibility Criteria for Personal Loans:
        - Indian residents aged between 21 years and 60 years of age.  
        - Salaried professionals with a minimum monthly income of ₹25,000 (as of [current year](https://www.kotak.com/current-criteria/)).  
        
        For more details on other loan types like home or business loans, visit the bank's website at [www.kotak.com](https://www.kotak.com).""",

        """1. Log in to the Kotak Mobile Banking App using MPIN, Fingerprint, or Net Banking credentials.
        2. On the home screen, your account balance will be displayed.
        3. If not visible, go to ‘Banking’ > ‘Account Details’ to view your balance.
        4. You can also check via WhatsApp Banking by sending “Balance” to 022-66006022.
        For further assistance, call 1860 266 2666.""",

        """Yes, transferring money online through Kotak Mahindra Bank is safe. We use 128-bit SSL encryption, two-factor authentication (OTP-based verification), and transaction alerts for security. 
        Your Net Banking and Mobile Banking access are protected with strong password policies, auto-logout features, and device binding for UPI transactions.
        For added safety, always log in via www.kotak.com, avoid sharing OTPs or passwords, and monitor SMS/email alerts. 
        If you suspect fraud, call 1800 209 0000 immediately."""
    ]
    
    evaluation_results, overall_score, is_trustworthy = evaluate_rag_pipeline(files, queries, ground_truths)
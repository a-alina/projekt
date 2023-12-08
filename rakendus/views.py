from calendar import prmonth
from django.shortcuts import render, redirect
import openai
from django.contrib import messages
from django.http import JsonResponse
from django.template import loader, Context
from django.template.loader import render_to_string
from .forms import UploadFileForm
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA 
from django.views.decorators.csrf import csrf_exempt
import os

first_question = "Hei /n ready to study? /n Please upload a file first"
conv_history = [ '\nQuestion: ' + first_question]

document_main = None

@csrf_exempt
def home(request):
    global document_main
    openai_api_key = "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"
    form = UploadFileForm(request.POST, request.FILES)


    # parsing text
    if len(request.FILES) != 0:
        pdf_reader = PdfReader(request.FILES['file'])
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # spliting text into chunks
        splitter = CharacterTextSplitter(
            separator='\n', 
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = splitter.split_text(text)

        # converting chunks into embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        document = FAISS.from_texts(chunks, embeddings)
        document_main = document
        
        

        return render(request, "home.html")
    
    if request.POST.get("question"):
        question = request.POST["question"]
        retriver = document_main.as_retriever(search_kwargs={"k": 5})
        llm = OpenAI(openai_api_key=openai_api_key)
        qa = RetrievalQA.from_chain_type(llm=llm, 
                retriever=retriver,
                chain_type="stuff")
        
        response = qa.run(query=question)
    

        return render(request, "home.html", {"response": response})


    return render(request, "home.html",
                  {"form": form})
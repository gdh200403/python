from django.shortcuts import render, HttpResponse

# Create your views here.

def index(request):
    return HttpResponse("Hello, Django!")

def user_list(request):
    return HttpResponse("User List")
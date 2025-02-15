from django.shortcuts import render

# Create your views here.

def drawing_board(request):
    return render(request, 'frontend/drawing_board.html')

from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from .forms import DatasetForm
import glob, os
from .predictionModel import HotelModel

hm = HotelModel()


def searchFile():
    os.chdir("C:/Users/Google Prep Oct 22/Music/HotelMLProject-master/HotelMLProject-master/hotelX/media/media")
    for file in glob.glob("*.csv"):
        print(file)
        return str(file)


def results(request):
    fileName = searchFile()
    # print(fileName)
    res = hm.decisionTree()
    print("working res")
    return render(request, 'layout/results.html')

    



def uploadFile(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        whichML = request.POST['dataset']
        
        if form.is_valid():
            form.save()
            # results()
            # redirect(results)
            if(whichML == "Decision Tree"):
                res = hm.decisionTree()
                res()
            elif (whichML == "Random Forest"):
                res = hm.randomForest()
                res()
            else:
                res = hm.linearRegression()
                res()
            print("2 http")
            return render(request, 'layout/results.html')
            # return HttpResponseRedirect("/") 
    else:
        form = DatasetForm()
        redirect(index)


    return render(request, "layout/upload.html", {
        "form": form
    })

# Create your views here.
def index(request):
    return render(request, 'layout/index.html')

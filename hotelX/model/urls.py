from django.contrib import admin
from django.urls import path
from .views import *
urlpatterns = [
    path('index/',index,name = "index"),
    path('',uploadFile,name = "uploadFile"),
    # path('',mlResult,name = "mlResult"),
]
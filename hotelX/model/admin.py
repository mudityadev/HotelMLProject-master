from django.contrib import admin

# Register your models here.
from .models import dataSet, MLModelCategory

admin.site.register(dataSet)
admin.site.register(MLModelCategory)
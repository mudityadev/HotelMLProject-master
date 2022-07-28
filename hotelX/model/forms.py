from django.forms import ModelForm
from .models import dataSet

class DatasetForm(ModelForm):
    class Meta: 
        model = dataSet
        fields = '__all__'
from django.db import models

# Category Model 
class MLModelCategory(models.Model):
    mlModel = models.CharField(max_length=20)

    @staticmethod
    def get_all_categories():
        return MLModelCategory.objects.all()
    def __str__(self):
        return self.mlModel

class dataSet(models.Model):
    testID =  models.AutoField(primary_key=True)
    dataset = models.CharField(max_length=30, null=True)
    datasetFile = models.FileField(upload_to="media",null=False)
    selectModel = models.ForeignKey(MLModelCategory, on_delete=models.CASCADE)

    def __str__(self):
        return self.dataset
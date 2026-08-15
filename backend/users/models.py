from django.db import models
from django.contrib.auth.models import User
from PIL import Image


# Extending User Model Using a One-To-One Link
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    avatar = models.ImageField(default='default.jpg', upload_to='profile_images')
    bio = models.TextField()

    def __str__(self):
        return self.user.username

    # resizing images
    def save(self, *args, **kwargs):
        super().save()

        img = Image.open(self.avatar.path)

        if img.height > 100 or img.width > 100:
            new_img = (100, 100)
            img.thumbnail(new_img)
            img.save(self.avatar.path)


class UserPredictModel(models.Model):
    # Nullable: 27 pre-existing rows from before user-tracking existed have no
    # owner and are left as orphaned legacy data (see docs/AUDIT.md) rather
    # than deleted or assigned to a guessed user.
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='predictions')
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    AverageAmountTransactionDay = models.FloatField()  # Assuming it's a decimal value
    TransactionAmount = models.FloatField()  # Assuming it's a decimal value
    Is_declined = models.CharField(max_length=100)  # Assuming this is a True/False field
    TotalNumberOfDeclinesDay = models.IntegerField()
    isForeignTransaction = models.CharField(max_length=100) # Assuming this is a True/False field
    isHighRiskCountry = models.CharField(max_length=100)  # Assuming this is a True/False field
    DailyChargebackAvgAmt = models.FloatField()  # Assuming it's a decimal value
    Six_MonthAvgChbkAmt = models.FloatField()  # Assuming it's a decimal value
    Six_MonthChbkFreq = models.IntegerField()  # Assuming it's an integer
    isFradulent = models.CharField(max_length=100) # Assuming this is a True/False field
    fraud_probability = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction: {self.isFradulent}"

    @property
    def Prediction(self):
        """Template-compatibility alias — app/model_db.html was written
        against the old in-memory dict's 'Prediction' key."""
        return self.isFradulent
    



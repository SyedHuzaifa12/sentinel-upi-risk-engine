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

    # --- Legacy (pre-Phase-4) fields: the 9-field RandomForest form's input +
    # output. Now nullable -- new rows created via the Phase 4 sandbox never
    # populate these, but old rows and their values are kept as-is, not
    # backfilled or deleted (see ml/legacy/README.md for the archived model
    # these used to feed).
    AverageAmountTransactionDay = models.FloatField(null=True, blank=True)
    TransactionAmount = models.FloatField(null=True, blank=True)
    Is_declined = models.CharField(max_length=100, null=True, blank=True)
    TotalNumberOfDeclinesDay = models.IntegerField(null=True, blank=True)
    isForeignTransaction = models.CharField(max_length=100, null=True, blank=True)
    isHighRiskCountry = models.CharField(max_length=100, null=True, blank=True)
    DailyChargebackAvgAmt = models.FloatField(null=True, blank=True)
    Six_MonthAvgChbkAmt = models.FloatField(null=True, blank=True)
    Six_MonthChbkFreq = models.IntegerField(null=True, blank=True)
    isFradulent = models.CharField(max_length=100, null=True, blank=True)
    fraud_probability = models.FloatField(null=True, blank=True)

    # --- Phase 4: raw event + v2 scoring service result. Nullable -- legacy
    # rows never populate these.
    payer_vpa = models.CharField(max_length=255, null=True, blank=True)
    payee_vpa = models.CharField(max_length=255, null=True, blank=True)
    amount = models.FloatField(null=True, blank=True)
    event_timestamp = models.DateTimeField(null=True, blank=True)
    action = models.CharField(max_length=20, null=True, blank=True)  # ALLOW | WARN | REVIEW | BLOCK
    risk_tier = models.CharField(max_length=20, null=True, blank=True)  # LOW | MEDIUM | HIGH | CRITICAL
    risk_score = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=100, null=True, blank=True)
    reason_codes = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction: {self.isFradulent or self.action}"

    @property
    def Prediction(self):
        """Template-compatibility alias — app/model_db.html was written
        against the old in-memory dict's 'Prediction' key."""
        return self.isFradulent


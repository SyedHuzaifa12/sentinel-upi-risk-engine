from django.contrib.auth.models import User
from django.db import models
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


class Decision(models.Model):
    """Read-only mirror of the `decisions` table (decisionlog/schema.py).

    `managed = False`: Django never creates, alters, or drops this table --
    it is 100% owned by `decisionlog/schema.py`'s `ensure_schema()`, exactly
    as it was before Phase 6 (the append-only triggers there are the actual
    enforcement; nothing here weakens that). This model exists purely so the
    Django review queue and admin can read/filter/order `decisions` through
    the ORM's query-building convenience, and so `ReviewLabel` below can
    declare a real `ForeignKey` to it for joins -- without Django ever
    touching that table's schema.
    """
    txn_id = models.TextField(unique=True)
    event = models.JSONField()
    feature_snapshot = models.JSONField()
    risk_score = models.FloatField()
    raw_score = models.FloatField()
    is_cold = models.BooleanField()
    action = models.CharField(max_length=20)
    risk_tier = models.CharField(max_length=20)
    reason_codes = models.JSONField()
    model_version = models.CharField(max_length=100)
    thresholds_version = models.CharField(max_length=100)
    feature_lib_version = models.CharField(max_length=100)
    latency_ms = models.FloatField()
    scored_at = models.DateTimeField()
    source = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = "decisions"
        ordering = ["-scored_at"]

    def __str__(self):
        return f"Decision {self.txn_id}: {self.action}"


class ReviewLabel(models.Model):
    """An analyst's disposition on a REVIEW/BLOCK decision. Writing a label
    NEVER touches `decisions` -- `decision` is a real FK for query
    convenience (joins, `decision.review_labels.all()`), but
    `db_constraint=False` means Django adds no actual Postgres FK constraint:
    `decisions`' schema stays exclusively decisionlog's to manage, and this
    table's own DDL is the only thing this migration touches.

    IMPORTANT (see DESIGN.md "Review queue and selective labelling bias"):
    only alerted transactions (REVIEW/BLOCK) ever reach this queue, so
    precision computed over reviewed cases is NOT the model's true
    precision -- it says nothing about false negatives that were never
    flagged. Never present it as such.
    """
    DISPOSITION_CHOICES = [
        ("CONFIRMED_FRAUD", "Confirmed fraud"),
        ("LEGIT", "Legitimate"),
        ("UNCLEAR", "Unclear"),
    ]

    decision = models.ForeignKey(
        Decision, on_delete=models.PROTECT, db_constraint=False, related_name="review_labels")
    disposition = models.CharField(max_length=20, choices=DISPOSITION_CHOICES)
    reviewer = models.ForeignKey(User, on_delete=models.PROTECT, related_name="review_labels")
    reviewed_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-reviewed_at"]

    def __str__(self):
        return f"ReviewLabel({self.decision.txn_id}, {self.disposition})"


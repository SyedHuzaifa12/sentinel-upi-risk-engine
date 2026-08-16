from datetime import timezone

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import redirect, render
from django.utils import timezone as dj_timezone

from ..models import Decision, ReviewLabel

PAGE_SIZE = 25


@login_required
def review_queue(request):
    if request.method == "POST":
        decision_id = request.POST.get("decision_id")
        disposition = request.POST.get("disposition")
        notes = request.POST.get("notes", "")
        if disposition not in dict(ReviewLabel.DISPOSITION_CHOICES):
            messages.error(request, "Invalid disposition.")
            return redirect("review_queue")
        try:
            decision = Decision.objects.get(pk=decision_id)
        except Decision.DoesNotExist:
            messages.error(request, "That decision no longer exists in the queue.")
            return redirect("review_queue")
        # Writing a label NEVER touches `decisions` -- only a new ReviewLabel
        # row is created. See Decision.Meta.managed = False / ReviewLabel's
        # db_constraint=False in models.py.
        ReviewLabel.objects.create(decision=decision, disposition=disposition, reviewer=request.user, notes=notes)
        messages.success(request, f"Labeled {decision.txn_id} as {disposition}.")
        return redirect("review_queue")

    queue = Decision.objects.filter(action__in=["REVIEW", "BLOCK"]).exclude(
        review_labels__isnull=False,
    ).order_by("-scored_at")

    paginator = Paginator(queue, PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))

    now = dj_timezone.now()
    ages_seconds = [(now - d.scored_at.replace(tzinfo=timezone.utc)).total_seconds() for d in queue]
    ages_seconds.sort()
    median_age_seconds = ages_seconds[len(ages_seconds) // 2] if ages_seconds else 0

    total_labels = ReviewLabel.objects.count()
    confirmed_fraud = ReviewLabel.objects.filter(disposition="CONFIRMED_FRAUD").count()
    reviewed_precision = (confirmed_fraud / total_labels) if total_labels else None

    return render(request, "app/review_queue.html", {
        "page": page,
        "queue_depth": queue.count(),
        "median_age_seconds": median_age_seconds,
        "reviewed_precision": reviewed_precision,
        "total_labels": total_labels,
        "disposition_choices": ReviewLabel.DISPOSITION_CHOICES,
    })

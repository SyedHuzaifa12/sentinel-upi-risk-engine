import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from feature_lib.event import make_ulid  # noqa: E402

from ..models import UserPredictModel
from ..services.prediction_service import RiskAPIUnavailable, predict_event


@login_required
def prediction_view(request):
    # A stable per-browser-session device id, so the sandbox form doesn't
    # make the user invent one -- see model.html's prefilled fields.
    if not request.session.session_key:
        request.session.save()
    device_id = request.session.get('sandbox_device_id')
    if not device_id:
        device_id = f"device-{uuid.uuid4().hex[:12]}"
        request.session['sandbox_device_id'] = device_id
    return render(request, 'app/model.html', {'device_id': device_id})


@login_required
def analyze_upi(request):
    if request.method == 'POST':
        now = datetime.now(timezone.utc)
        event = {
            # Minted server-side -- a real payment app supplies these, a
            # single-event sandbox form isn't where a transaction id or
            # timestamp should come from.
            'txn_id': make_ulid(now, np.random.default_rng()),
            'timestamp': now.isoformat(),
            'payer_vpa': request.POST.get('payer_vpa', ''),
            'payee_vpa': request.POST.get('payee_vpa', ''),
            'amount': request.POST.get('amount'),
            'txn_type': request.POST.get('txn_type'),
            'initiation_mode': request.POST.get('initiation_mode'),
            'device_id': request.POST.get('device_id', ''),
            'payer_bank': request.POST.get('payer_bank', ''),
            'payee_bank': request.POST.get('payee_bank', ''),
            'payer_account_age_days': request.POST.get('payer_account_age_days'),
        }
        try:
            event['amount'] = float(event['amount'])
            event['payer_account_age_days'] = int(event['payer_account_age_days'])
        except (TypeError, ValueError):
            messages.error(request, 'Could not analyze this transaction: amount and account age must be numbers.')
            return redirect(to='prediction')

        try:
            result = predict_event(event)
        except RiskAPIUnavailable:
            messages.error(
                request,
                'The risk scoring service is unavailable right now. Please try again shortly.',
            )
            return redirect(to='prediction')

        record = UserPredictModel.objects.create(
            user=request.user,
            payer_vpa=event['payer_vpa'],
            payee_vpa=event['payee_vpa'],
            amount=event['amount'],
            event_timestamp=now,
            action=result['action'],
            risk_tier=result['risk_tier'],
            risk_score=result['risk_score'],
            model_version=result['model_version'],
            reason_codes=result['reason_codes'],
        )

        return render(request, 'app/result.html', {
            'action': result['action'],
            'risk_tier': result['risk_tier'],
            'risk_score': result['risk_score'],
            'reason_codes': result['reason_codes'],
            'is_cold': result['is_cold'],
            'model_version': result['model_version'],
            'thresholds_version': result['thresholds_version'],
            'timestamp': record.created_at,
            'reference_id': record.id,
        })

    return redirect(to='prediction')


@login_required
def model_db_view(request):
    user_db = UserPredictModel.objects.filter(user=request.user)
    return render(request, 'app/model_db.html', {'user_db': user_db})

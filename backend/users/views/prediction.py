from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from ..models import UserPredictModel
from ..services.prediction_service import InvalidInputError, predict


@login_required
def prediction_view(request):
    return render(request, 'app/model.html')


@login_required
def analyze_upi(request):
    if request.method == 'POST':
        data = {
            'AverageAmountTransactionDay': request.POST.get('AverageAmountTransactionDay'),
            'TransactionAmount': request.POST.get('TransactionAmount'),
            'Is_declined': request.POST.get('Is_declined'),
            'TotalNumberOfDeclinesDay': request.POST.get('TotalNumberOfDeclinesDay'),
            'isForeignTransaction': request.POST.get('isForeignTransaction'),
            'isHighRiskCountry': request.POST.get('isHighRiskCountry'),
            'DailyChargebackAvgAmt': request.POST.get('DailyChargebackAvgAmt'),
            'Six_MonthAvgChbkAmt': request.POST.get('Six_MonthAvgChbkAmt'),
            'Six_MonthChbkFreq': request.POST.get('Six_MonthChbkFreq'),
        }

        try:
            result = predict(data)
        except InvalidInputError as exc:
            messages.error(request, f'Could not analyze this transaction: {exc}')
            return redirect(to='prediction')

        UserPredictModel.objects.create(
            user=request.user,
            AverageAmountTransactionDay=data['AverageAmountTransactionDay'],
            TransactionAmount=data['TransactionAmount'],
            Is_declined=data['Is_declined'],
            TotalNumberOfDeclinesDay=data['TotalNumberOfDeclinesDay'],
            isForeignTransaction=data['isForeignTransaction'],
            isHighRiskCountry=data['isHighRiskCountry'],
            DailyChargebackAvgAmt=data['DailyChargebackAvgAmt'],
            Six_MonthAvgChbkAmt=data['Six_MonthAvgChbkAmt'],
            Six_MonthChbkFreq=data['Six_MonthChbkFreq'],
            isFradulent=result.label,
            fraud_probability=result.fraud_probability,
        )

        return render(request, 'app/result.html', {
            'prediction': result.label,
            'fraud_probability': round(result.fraud_probability * 100, 1),
            'top_features': result.top_features,
        })

    return redirect(to='prediction')


@login_required
def model_db_view(request):
    user_db = UserPredictModel.objects.filter(user=request.user)
    return render(request, 'app/model_db.html', {'user_db': user_db})

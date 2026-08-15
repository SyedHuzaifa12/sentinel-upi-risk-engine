from django.shortcuts import render


def Basic_report(request):
    return render(request, 'app/Basic_report.html')


def Metrics_report(request):
    return render(request, 'app/Metrics_report.html')

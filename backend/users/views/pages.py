from django.shortcuts import render

from ..models import Profile


def profile_list(request):
    # Fetch all profile objects from the database
    profiles = Profile.objects.all()

    # Pass the profiles data to the template
    return render(request, 'app/profile_list.html', {'profiles': profiles})


def awareness_page(request):
    return render(request, 'app/awareness.html')

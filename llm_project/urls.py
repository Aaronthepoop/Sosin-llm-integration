from django.urls import path
from integration import views  # Updated import

urlpatterns = [
    # Your URL patterns here
    path('', views.index, name='index'),  # Example view
]


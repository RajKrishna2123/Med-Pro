"""medpro URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name="home"),
    path('form1/', views.ocrinp, name="home"),
    path('form1/ocr', views.ocr, name="home"),
    path('threeopt/liv', views.livinp,name="livinp"),
    path('threeopt/',views.threeopt,name="Three Option"),
    path('threeopt/liv_pred' ,views.liv, name="liv"),
    path('threeopt/heart',views.heart,name="Heart Input"),
    path('form1/OCRsubmit',views.OCRsubmit,name="OCR Submit"),
    path('cancer/brain',views.brain,name="cancer"),
    path('cancer/breast',views.breast,name="cancer"),
    path('cancer/all',views.all,name="cancer"),
    path('cancer/lym',views.lymph,name="cancer"),
    path('cancer/kidney',views.kidney,name="cancer"),
    path('cancer/cervical',views.cervical,name="cancer"),
    path('cancer/lung',views.lung,name="cancer"),
    path('cancer/oral',views.oral,name="cancer"),
    path('cancer/',views.cancer,name="cancer"),
    path('cancer/brain_pred',views.brain_pred,name="cancer"),
    path('cancer/breast_pred',views.breast_pred,name="cancer"),
    path('cancer/all_pred',views.all_pred,name="cancer"),
    path('cancer/lymph_pred',views.lymph_pred,name="cancer"),
    path('cancer/kidney_pred',views.kidney_pred,name="cancer"),
    path('cancer/cervical_pred',views.cervical_pred,name="cancer"),
    path('cancer/lung_pred',views.lung_pred,name="cancer"),
    path('cancer/oral_pred',views.oral_pred,name="cancer"),
    path('threeopt/skinpred',views.skinpred,name="Skin Prediction"),
    path('segment',views.segment,name="Tumor Segmentation"),
    path('segmentation',views.segmentation,name="Tumor Segmentation"),
    ]

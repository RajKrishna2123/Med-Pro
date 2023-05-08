from django.core.files.storage import default_storage
from django.conf import settings
from . import urls
from django.shortcuts import render
import os 
import requests
import cv2
import numpy as np
from django.http import HttpResponse
import numpy as np
import joblib
from numba import cuda
import csv

textarr=" "
def home(request):
    return render (request, "home.html")
def ocrinp(request):
    return render(request, "form.html")
def ocr(request):
    extracted_text = ''
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        save_dir = 'temp'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(settings.BASE_DIR, 'static/temp', img_file.name)
        file_name = default_storage.save(file_path, img_file)

        url = "https://api.ocr.space/parse/image"
        payload = {
            "apikey": "K84776460288957",
            "language": "eng",
            "isOverlayRequired": False
        }
        save_path='static/temp/a.jpg'
        with open(save_path, "rb") as f:
            r = requests.post(url, files={"a.jpg": f}, data=payload)

        if r.status_code == 200:
            response = r.json()
            if response["IsErroredOnProcessing"] == False:
                extracted_text = response["ParsedResults"][0]["ParsedText"]
                print(extracted_text)
            else:
                error_message = response["ErrorMessage"]
                print("OCR Error: " + error_message)
        else:
            print("Request Error")
        f="D:\\medpro\\static\\temp\\data.csv"
        with open(f, 'w') as csvfile:
            csvw=csv.writer(csvfile)
            csvw.writerow(extracted_text)
        
        c=extracted_text.split("\n")
        textarr=c
        processed_text=[]
        for i in c:
            
            processed_text.append(i)
    return render(request, "output2.html",{"raw":processed_text})

def livinp(request):
    return render(request, 'form2.html')
def liv(request):
    knn=joblib.load('KNNClassifier_93')
    dtr=joblib.load('dtr_99')
    lrc=joblib.load('LRclassifier_71')
    rfc=joblib.load('rfc_99')
    xgb=joblib.load('xgb_995')
    
    Age_of_the_patient=float(request.POST.get('Age_of_the_patient','default'))
    Gender_of_the_patient=request.POST.get('Gender_of_the_patient','default')
    if Gender_of_the_patient=="Male":
        Gender_of_the_patient=1.0
    elif Gender_of_the_patient=="Female":
        Gender_of_the_patient=0.0
    else:
        Gender_of_the_patient=0.0
    Direct_Bilirubin=float(request.POST.get('Direct_Bilirubin','default'))
    Alkphos_Alkaline_Phosphotase=float(request.POST.get('Alkphos_Alkaline_Phosphotase','default'))
    Sgpt_Alamine_Aminotransferase=float(request.POST.get('Sgpt_Alamine_Aminotransferase','default'))
    Total_Protiens=float(request.POST.get('Total_Protiens','default'))
    A_G_Ratio_Albumin_and_Globulin_Ratio=float(request.POST.get('A_G_Ratio_Albumin_and_Globulin_Ratio','default'))

    if Gender_of_the_patient==1.0:
        prediction_array=[[1.0,0.0,0.0,Age_of_the_patient,Direct_Bilirubin,Alkphos_Alkaline_Phosphotase,
                      Sgpt_Alamine_Aminotransferase,Total_Protiens,A_G_Ratio_Albumin_and_Globulin_Ratio]]
    elif Gender_of_the_patient==0.0:
        prediction_array=[[0.0,1.0,0.0,Age_of_the_patient,Direct_Bilirubin,Alkphos_Alkaline_Phosphotase,
                      Sgpt_Alamine_Aminotransferase,Total_Protiens,A_G_Ratio_Albumin_and_Globulin_Ratio]]
    else:
        prediction_array=[[0.0,0.0,0.0,Age_of_the_patient,Direct_Bilirubin,Alkphos_Alkaline_Phosphotase,
                      Sgpt_Alamine_Aminotransferase,Total_Protiens,A_G_Ratio_Albumin_and_Globulin_Ratio]]
    
    rawdata={'knn':knn.predict(prediction_array),'dtr':dtr.predict(prediction_array),
             'lrc':lrc.predict(prediction_array),'rfc':rfc.predict(prediction_array),
             'xgb':xgb.predict(prediction_array)}
    
    if rawdata["dtr"]==1:
        return render (request, 'output1.html', {'raw':rawdata})
    else:
        return render (request, 'output_.html', {'raw':rawdata})


    
# def contact(request):
#     if(request.method=="POST"):
#         name=request.POST.get("name")
#         email=request.POST.get("email")

def OCRsubmit(request):
    if request.method=="POST":
        text=request.POST.get("text")
        
        

    return render(request,"home.html")
def threeopt(request):
    return render(request,"three.html")
def heart(request):
    return render(request,"heart.html")

def cancer(request):
    return render(request,"menu.html")
def brain(request):
    return render(request,"form_brain.html")
def breast(request):
    return render(request,"form_breast.html")
def all(request):
    return render(request,"form_all.html")
def lymph(request):
    return render(request,"form_lymph.html")
def kidney(request):
    return render(request,"form_kidney.html")
def cervical(request):
    return render(request,"form_cervical.html")
def lung(request):
    return render(request,"form_lung.html")
def oral(request):
    return render(request,"form_oral.html")

def image_handle(x):
    img=cv2.imread(x)
    # Resize the image to 64x64 pixels
    img_resized = cv2.resize(img, (128, 128))
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def cancer_pred(request):
    return render(request,"menu.html")

def brain_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\brain', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('brain_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0:
        raw="brain_glioma"
    elif prediction[0][0]==0.0 and prediction[0][1]==1.0 and prediction[0][2]==0.0:
        raw="brain_menin"
    else:
        raw="brain_tumor"
        
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw})

def breast_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\breast', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('breast_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0:
        raw="breast_benign"
    else:
        raw="breast_malignant"
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw})

def all_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\all', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('All_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0 and prediction[0][3]==0.0:
        raw="all_begnign"
    elif prediction[0][0]==0.0 and prediction[0][1]==1.0 and prediction[0][2]==0.0 and prediction[0][3]==0.0:
        raw="all_early"
    elif prediction[0][0]==0.0 and prediction[0][1]==0.0 and prediction[0][2]==1.0 and prediction[0][3]==0.0:
        raw="all_pre"
    else:
        raw="all_pro"
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw})

def lymph_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\lymph', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('lymph_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0:
        raw="lymph_cll"
    elif prediction[0][0]==0.0 and prediction[0][1]==1.0 and prediction[0][2]==0.0:
        raw="lymph_fl"
    else:
        raw="lymph_mcl"
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw})

def kidney_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\kidney', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('Kidney_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0:
        raw="kidney_normal"
    else:
        raw="kidney_tumor"
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw})

def cervical_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\cervical', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('cervical.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0 and prediction[0][3]==0.0 and prediction[0][4]==0.0:
        raw="cervix_dyk"
    elif prediction[0][0]==0.0 and prediction[0][1]==1.0 and prediction[0][2]==0.0 and prediction[0][3]==0.0 and prediction[0][4]==0.0:
        raw="cervix_koc"
    elif prediction[0][0]==0.0 and prediction[0][1]==0.0 and prediction[0][2]==1.0 and prediction[0][3]==0.0 and prediction[0][4]==0.0:
        raw="cervix_mep"
    elif prediction[0][0]==0.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0 and prediction[0][3]==1.0 and prediction[0][4]==0.0:
        raw="cervix_pab"
    else:
        raw="cervix_sfi"
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw}) 

def lung_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\lung', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('lung_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0 and prediction[0][3]==0.0 and prediction[0][4]==0.0:
        raw="colon_aca"
    elif prediction[0][0]==0.0 and prediction[0][1]==1.0 and prediction[0][2]==0.0 and prediction[0][3]==0.0 and prediction[0][4]==0.0:
        raw="colon_bnt"
    elif prediction[0][0]==0.0 and prediction[0][1]==0.0 and prediction[0][2]==1.0 and prediction[0][3]==0.0 and prediction[0][4]==0.0:
        raw="lung_aca"
    elif prediction[0][0]==0.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0 and prediction[0][3]==1.0 and prediction[0][4]==0.0:
        raw="lung_bnt"
    else:
        raw="lung_scc"
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw,"path":img_path})
def oral_pred(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\oral', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('oral_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    device = cuda.get_current_device()
    device.reset()
    if prediction[0][0]==1.0 and prediction[0][1]==0.0:
        raw="oral_normal"
    else:
        raw="oral_scc"
    raw.lstrip()
    raw.rstrip()
    return render(request,"disout.html",{"raw":raw}) 
def skinpred(request):
    return render(request,"form_skin.html")

def segment(request):
    return render(request,"form_segmentation.html")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
def segmentation(request):
    if request.method == 'POST' and request.FILES['img_file']:
        img_file = request.FILES['img_file']
        img_path = os.path.join('static\\temp\\brain', img_file.name)
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)
        name=img_file.name
    
    temp=image_handle(img_path)
    import tensorflow as tf 
    loaded_model = tf.keras.models.load_model('brain_cancer.h5')
    prediction = loaded_model.predict(temp)
    print(prediction)
    
    if prediction[0][0]==1.0 and prediction[0][1]==0.0 and prediction[0][2]==0.0:
        raw="glioma"
        return render(request,"notumor.html")
    elif prediction[0][0]==0.0 and prediction[0][1]==1.0 and prediction[0][2]==0.0:
        raw="menin"
        return render(request,"notumor.html")
    else:
        image_file = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image_file, channels=3)

        # Define the sliding window parameters
        window_size = (128, 128)
        stride = (64, 64)

        # Create a list to store the patches
        patches = []

        # Create a list to store the patch locations
        locations = []

        # Slide the window over the image
        for i in range(0, image.shape[0]-window_size[0], stride[0]):
            for j in range(0, image.shape[1]-window_size[1], stride[1]):
                patch = tf.image.crop_to_bounding_box(image, i, j, window_size[0], window_size[1])
                patches.append(patch)
                locations.append((i, j))

        # Convert the patches to a batch
        batch = tf.stack(patches)

        # Make predictions on the batch using the CNN model
        predictions = loaded_model.predict(batch)

        # Find the index of the patch with the highest tumor probability
        max_index = np.argmax(predictions[:,2])

        # Get the location of the patch with the highest tumor probability
        max_location = locations[max_index]

        # Create a bounding box over the patch with the highest tumor probability
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = cv2.rectangle(image_array, (max_location[1], max_location[0]), (max_location[1]+window_size[1], max_location[0]+window_size[0]), (0, 0, 255), 2)
        image = tf.keras.preprocessing.image.array_to_img(image_array)

        # Save the image in a certain location
        pred_img_path = os.path.join('static\\pred', name)
        image.save(pred_img_path)
        n=os.path.join("static/pred",name)
        device = cuda.get_current_device()
        device.reset()
    return render(request,"segimg.html",{"raw":n})
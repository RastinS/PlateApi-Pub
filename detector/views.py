import logging
from django import views
from rest_framework.views import APIView
from django.shortcuts import render
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from django.http import JsonResponse
from rest_framework import status
import urllib
import cv2
import numpy as np

from .recognizer.detect import detectPhoto


class FileUploadView(APIView):
    parser_classes = ( MultiPartParser, FormParser, JSONParser)
    logger = logging.getLogger('imageUpload')

    def post(self, request, *args, **kwargs):
        data = {"success": False}
        try:
            if request.FILES.get("plateImage", None) is not None:
                plateImage = self._grab_image(stream=request.FILES["plateImage"])
            elif request.POST.get('plateUrl') is not None:
                url = request.POST.get("plateUrl", None)
                plateImage = self._grab_image(url=url)
            elif request.POST.get('plateAddress', None) is not None:
                plateImageAddress = request.POST.get('plateAddress', None)
                plateImage = self._grab_image(path=plateImageAddress)
            else:
                return JsonResponse(status=status.HTTP_400_BAD_REQUEST, data={'Message': 'Data was not provided'})

            results = detectPhoto(plateImage, asJSON=False)

            return JsonResponse(results, safe=False, status=status.HTTP_201_CREATED)
        except Exception as e:
            self.logger.exception("\n")
            return JsonResponse(status=status.HTTP_500_INTERNAL_SERVER_ERROR, data={'Message': 'Internal server error in detecting plate image'})

    @staticmethod
    def _grab_image(path=None, stream=None, url=None):
        print('path: {}  -  stream: {}  -  url: {}'.format(path, stream, url))
        if path is not None:
            image = cv2.imread(path)
        else:	
            if url is not None:
                resp = urllib.urlopen(url)
                data = resp.read()
            elif stream is not None:
                data = stream.read()

            image = np.asarray(bytearray(data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
        return image
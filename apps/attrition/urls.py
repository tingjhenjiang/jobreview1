from django.urls import re_path,include
from rest_framework.routers import DefaultRouter

from apps.attrition.views import EndpointViewSet
from apps.attrition.views import MLAlgorithmViewSet
from apps.attrition.views import MLAlgorithmStatusViewSet
from apps.attrition.views import MLRequestViewSet

router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    re_path(r"^api/attrition/v1/", include(router.urls)),
]
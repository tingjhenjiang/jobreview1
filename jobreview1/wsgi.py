"""
WSGI config for jobreview1 project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jobreview1.settings')

application = get_wsgi_application()
# ML registry
import inspect
from apps.attritionml.registry import MLRegistry
from apps.attritionml.mlmodeldefinition import AttritionClassifier

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    clf = AttritionClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="attrition_classifier",
                            algorithm_object=clf,
                            algorithm_name="logisticRegL2",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="TJ",
                            algorithm_description="RlogisticRegL2 with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(AttritionClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
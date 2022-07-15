from django.test import TestCase
import inspect

from apps.attritionml.mlmodeldefinition import AttritionClassifier
from apps.attritionml.registry import MLRegistry

class MLTests(TestCase):
    def test_algorithm(self):
        import pandas as pd
        input_data = {
            "Age":41,
            "BusinessTravel":"Travel_Rarely",
            "DailyRate":1102,
            "Department":"Sales",
            "DistanceFromHome":1,
            "Education":2,
            "EducationField":"Life Sciences",
            "EmployeeCount":1,
            "EmployeeNumber":1,
            "EnvironmentSatisfaction":2,
            "Gender":"Female",
            "HourlyRate":94,
            "JobInvolvement":3,
            "JobLevel":2,
            "JobRole":"Sales Executive",
            "JobSatisfaction":4,
            "MaritalStatus":"Single",
            "MonthlyIncome":5993,
            "MonthlyRate":19479,
            "NumCompaniesWorked":8,
            "Over18":"Y",
            "OverTime":"Yes",
            "PercentSalaryHike":11,
            "PerformanceRating":3,
            "RelationshipSatisfaction":1,
            "StandardHours":80,
            "StockOptionLevel":0,
            "TotalWorkingYears":8,
            "TrainingTimesLastYear":0,
            "WorkLifeBalance":1,
            "YearsAtCompany":6,
            "YearsInCurrentRole":4,
            "YearsSinceLastPromotion":0,
            "YearsWithCurrManager":5
        }
        my_alg = AttritionClassifier()
        response = my_alg.compute_prediction(input_data)
        print(response)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "attrition_classifier"
        algorithm_object = AttritionClassifier()
        algorithm_name = "LogisticRegL2"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(AttritionClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
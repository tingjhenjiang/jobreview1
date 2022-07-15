from django.test import TestCase

# Create your tests here.
from django.test import TestCase
from rest_framework.test import APIClient

class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
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
        classifier_url = "/api/attrition/v1/attrition_classifier/predict"
        response = client.post(classifier_url, input_data, format='json')
        print(response)
        print(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)
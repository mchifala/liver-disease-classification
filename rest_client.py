import requests
import argparse
from jsonpickle import encode
from jsonpickle import decode


def main(host_ip, patient_id, feature_vector):
    """
    This function makes an HTTP POST call to the "/prediction"
    API endpoint to get a prediction of liver vs. no liver disease
    for a user-input feature vector.

    Parameters:
    - host_ip(str): The host IP address for the API endpoint
    - patient_id(str): The patient ID number or identifier
    - feature_vector(str): A feature vector of 11 features for patient

    Returns:
    - None; the status code and response JSON are printed

    """
    address = "http://"+host_ip+":5000"+"/prediction"
    headers = {"content-type": "application/json"}
    data = {"id": patient_id, "features": feature_vector}
    response = requests.post(address, data=encode(data), headers=headers)
    print("Status:", response)
    print("Result:", decode(response.text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="REST API")

    parser.add_argument("host_ip",
                        type=str,
                        help="Address of the server ex. localhost")

    parser.add_argument("patient_id",
                        type=str,
                        help="The patient ID number")

    parser.add_argument("feature_vector",
                        type=str,
                        nargs="+",
                        help="A feature vector of 11 features for patient")

    args = parser.parse_args()
    main(args.host_ip, args.patient_id, args.feature_vector)

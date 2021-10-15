# !pip install mlflow
# !pip install minio
# !pip install boto3
# !pip install scikit-learn==0.24.2
# !pip install tensorflow==2.6.0
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])#,"--user"
install('tensorflow==2.6.0')
install('mlflow')
install('minio')
install('jinja2')
# install('openshift')
install('openshift-client')
install('boto3')


# !pip show mlflow
# !pip show minio
# !pip show boto3
# !pip show scikit-learn
# !pip show openshift-client

import os
import mlflow
from minio import Minio
import openshift as oc
from jinja2 import Template
import tensorflow as tf
os.environ['MLFLOW_S3_ENDPOINT_URL']='http://minio-ml-workshop:9000'
os.environ['AWS_ACCESS_KEY_ID']='minio'
os.environ['AWS_SECRET_ACCESS_KEY']='minio123'
os.environ['AWS_REGION']='us-east-1'
os.environ['AWS_BUCKET_NAME']='mlflow'
os.environ['MODEL_NAME'] = 'lstmv7'
os.environ['MODEL_VERSION'] = '1'
# os.environ['OPENSHIFT_CLIENT_PYTHON_DEFAULT_OC_PATH'] = '/tmp/oc'

HOST = "http://mlflow:5500"

model_name = os.environ["MODEL_NAME"]
model_version = os.environ["MODEL_VERSION"]
build_name = f"seldon-model-{model_name}-v{model_version}"

def get_s3_server():
    minioClient = Minio('minio-ml-workshop:9000',
                    access_key='minio',
                    secret_key='minio123',
                    secure=False)

    return minioClient


def init():
    mlflow.set_tracking_uri(HOST)
    print(HOST)
    # Set the experiment name...
    #mlflow_client = mlflow.tracking.MlflowClient(HOST)

    
def download_artifacts():
    print("retrieving model metadata from mlflow...")
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    print(model)
    
    run_id = model.metadata.run_id
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    
    print("initializing connection to s3 server...")
    minioClient = get_s3_server()

#     artifact_location = mlflow.get_experiment_by_name('rossdemo').artifact_location
#     print("downloading artifacts from s3 bucket " + artifact_location)

    data_file_model = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/model.h5", "model.h5")
    # TODO  # REPLACE IT WITH FOR LOOP TO AUTOMATICA
    data_file_tokenizer = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/tokenizer.pkl", "tokenizer.pkl")
    data_file_labelencoder = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/labelencoder.pkl", "labelencoder.pkl")
    data_file_tokenizer = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/requirements.txt", "requirements.txt")
    #Using boto3 Download the files from mlflow, the file path is in the model meta
    #write the files to the file system
    print("download successful")
    
    return run_id
    
        
init()
run_id = download_artifacts()

print("Start OCP things...")

server = "https://" + os.environ["KUBERNETES_SERVICE_HOST"] + ":" + os.environ["KUBERNETES_SERVICE_PORT"] 
print(server)

# #build from source Docker file
# with open('/var/run/secrets/kubernetes.io/serviceaccount/token', 'r') as file:
#     token = file.read()
# print(f"Openshift Token{token}")
token = 'eyJhbGciOiJSUzI1NiIsImtpZCI6ImF3d2ZabEZ1VG4zNzR3dV9tV0NKZUlpUF8zaWpqUGFnQVFVbU9vbkk5V28ifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJhbnotbWwiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoiZGVmYXVsdC10b2tlbi1obm1yciIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiZThjODUzYmUtY2JmOC00MTA3LWE1NzMtZTc1OTFjMTY3MjUzIiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OmFuei1tbDpkZWZhdWx0In0.PZm7Typek4-xyPqhdB9wwPqJ8NAyoSzX7KlHJgA3cMNnJX4zfYql2msDOSF1WSChHCDI8MpxKSmrI5DJD1IOCJOGbORr-aYqbMSoYdo2lGlSDSxJRUnlL-9rph6chclR_FCctEZbq4SpZlxG03relvlM3913qn1o9CXgApDRIUAsPIBc2Ug6q7nIS6ifhz6S8yJ9oIhnJvs7PgGUJ0sf6z11TCN7kTjFIg8dObI4UFXRrKspPNqXdwRV94Tn9hVljihKutSbNEcT20DEXGkXqSdqrIKQjTVZ_yqMl2rQ6LV9_FKdemsqK21z5VxzNP9yIM0kX3M-Z8bn4GPkK4Y9Gw'
# #/var/run/secrets/kubernetes.io/serviceaccount
# with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r') as namespace:
#     project = namespace.read()
# print(f"Project name: {project}")
project = 'anz-ml'

#build from source Docker file
with oc.apis_server(erver):
    with oc.token(token):
        with oc.project(project), oc.timeout(10*60):
            print('OpenShift client version: {}'.format(oc.get_client_version()))
            #print('OpenShift server version: {}'.format(oc.get_server_version()))

            build_config = oc.selector(f"bc/{build_name}").count_existing() #.object
            print(oc.get_project_name())
            print(build_config)
            if build_config == 0:
                oc.new_build(["--strategy", "docker", "--binary", "--docker-image", "registry.access.redhat.com/ubi8/python-38:1-71", "--name", build_name ])
            else:
                build_details = oc.selector(f"bc/{build_name}").object()
                print(build_details.as_json())

            print("Starting Build and Wiating.....")
            build_exec = oc.start_build([build_name, "--from-dir", ".", "--follow", "--build-loglevel", "10"])# docker build and push
            print("Build Finished")
            status = build_exec.status()
            print(status)
            for k, v in oc.selector([f"bc/{build_name}"]).logs(tail=500).items():
                print('Build Log: {}\n{}\n\n'.format(k, v))

            seldon_deploy = oc.selector(f"SeldonDeployment/{build_name}").count_existing()

            experiment_id = mlflow.get_run(run_id).info.experiment_id

            if seldon_deploy == 0:
                template_data = {"experiment_id": run_id, "model_name": model_name, "image_name": build_name, "project": project}
                applied_template = Template(open("SeldonDeploy.yaml").read())
                print(applied_template.render(template_data))
                oc.create(applied_template.render(template_data))
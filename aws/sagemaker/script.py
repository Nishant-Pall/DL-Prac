import sagemaker
import boto3

print(sagemaker.__version__)


sess = sagemaker.Session()

# !aws iam list-roles | grep AmazonSageMaker-ExecutionRole
role = 'arn:aws:iam::363886905145:role/service-role/AmazonSageMaker-ExecutionRole-20221115T200939'
print(sess)

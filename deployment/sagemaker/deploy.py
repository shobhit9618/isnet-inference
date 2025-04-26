#!/usr/bin/env python
"""
Deploy the model to AWS SageMaker.
"""
import os
import sys
import argparse
import logging
import time
import json

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy to SageMaker')
    
    parser.add_argument(
        '--image-uri',
        required=True,
        help='The URI of the Docker image to use'
    )
    parser.add_argument(
        '--model-data',
        required=True,
        help='The S3 path to the model data'
    )
    parser.add_argument(
        '--endpoint-name',
        required=True,
        help='The name of the endpoint to create or update'
    )
    parser.add_argument(
        '--instance-type',
        default='ml.g4dn.xlarge',
        help='The instance type to use for the endpoint'
    )
    parser.add_argument(
        '--instance-count',
        type=int,
        default=1,
        help='The number of instances to use for the endpoint'
    )
    parser.add_argument(
        '--region',
        help='The AWS region to use'
    )
    
    return parser.parse_args()


def get_execution_role(region=None):
    try:
        session = boto3.Session(region_name=region) if region else boto3.Session()
        iam_client = session.client('iam')
        
        response = iam_client.get_role(RoleName='AmazonSageMaker-ExecutionRole')
        return response['Role']['Arn']
    except Exception as e:
        logger.error(f"Error getting SageMaker execution role: {e}")
        raise


def deploy_model(args):
    try:
        region = args.region or boto3.Session().region_name
        sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
        
        role = get_execution_role(region)
        
        model = Model(
            image_uri=args.image_uri,
            model_data=args.model_data,
            role=role,
            name=f"{args.endpoint_name}-model",
            sagemaker_session=sagemaker_session
        )
        
        # Check if the endpoint exists
        exists = False
        client = boto3.client('sagemaker', region_name=region)
        
        try:
            client.describe_endpoint(EndpointName=args.endpoint_name)
            exists = True
        except client.exceptions.ClientError:
            exists = False
        
        # Deploy the model
        if exists:
            logger.info(f"Updating existing endpoint: {args.endpoint_name}")
            
            config_name = f"{args.endpoint_name}-config-{int(time.time())}"
            
            client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': f"{args.endpoint_name}-model",
                        'InitialInstanceCount': args.instance_count,
                        'InstanceType': args.instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            # Update the endpoint
            client.update_endpoint(
                EndpointName=args.endpoint_name,
                EndpointConfigName=config_name
            )
            
            logger.info(f"Waiting for endpoint update: {args.endpoint_name}")
            waiter = client.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=args.endpoint_name)
            
        else:
            logger.info(f"Creating new endpoint: {args.endpoint_name}")
            
            # Deploy model to endpoint
            predictor = model.deploy(
                endpoint_name=args.endpoint_name,
                instance_type=args.instance_type,
                initial_instance_count=args.instance_count,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            logger.info(f"Waiting for endpoint creation: {args.endpoint_name}")
            predictor.wait()
        
        logger.info(f"Endpoint deployed successfully: {args.endpoint_name}")
        
        return {
            'endpoint_name': args.endpoint_name,
            'model_name': f"{args.endpoint_name}-model",
            'instance_type': args.instance_type,
            'instance_count': args.instance_count,
            'image_uri': args.image_uri,
            'model_data': args.model_data,
            'region': region
        }
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise


def main():
    try:
        args = parse_args()
        result = deploy_model(args)
        logger.info(f"Deployment successful: {json.dumps(result)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in deployment: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 
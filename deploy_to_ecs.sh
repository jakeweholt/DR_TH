docker build -t loan_default_predictor .
eval $(aws ecr get-login --no-include-email --profile bsa | sed 's|https://||');
docker tag loan_default_predictor:latest $ECR_LOAN_DEFAULT_ADDRESS;
docker push $ECR_LOAN_DEFAULT_ADDRESS;
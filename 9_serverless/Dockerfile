FROM agrigorev/model-2024-hairstyle:v3

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY lambda_function_hw9.py .

CMD [ "lambda_function_hw9.lambda_handler" ]

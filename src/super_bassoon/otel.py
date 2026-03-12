import asyncio
import os
import logging
from opentelemetry import _logs
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LoggingHandler,
)
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import set_tracer_provider
from opentelemetry.sdk.trace import TracerProvider


class Otel(logging.Logger):
    def __init__(self, service_name: str, host: str):
        super().__init__(service_name)
        
        self.setLevel(logging.DEBUG)
        
        resource = Resource.create({
            SERVICE_NAME: service_name,
            "deployment.environment": os.environ.get("ENV", "development"),
        })
        
        tracer_provider = TracerProvider(resource=resource)
        set_tracer_provider(tracer_provider)
        
        logger_provider = LoggerProvider(resource=resource)
        _logs.set_logger_provider(logger_provider)
        
        endpoint = f"{host}/otlp/v1/logs"
        
        exporter = OTLPLogExporter(endpoint=endpoint)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )
        
        otel_handler = LoggingHandler(logger_provider=logger_provider)
        otel_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.addHandler(otel_handler)

    def validate_connection(self) -> bool:
        self.info("OTEL connection test - validate before proceeding")
        
        import time
        time.sleep(1)
        
        return True


async def main():
    host = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://192.168.68.222:3100")
    
    logger = Otel(service_name="super-bassoon", host=host)
    
    logger.info("Hello world")
    
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())

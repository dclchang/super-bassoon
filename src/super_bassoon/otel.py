import asyncio
from typing import Optional
import os
import logging
import base64
from opentelemetry import _logs
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LoggingHandler,
)
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import set_tracer_provider
from opentelemetry.sdk.trace import TracerProvider
from super_bassoon.op import get_secret


class Otel(logging.Logger):
    def __init__(self, service_name: str, host: str, instance_id: str, api_key: str):
        super().__init__(service_name)

        self.setLevel(logging.DEBUG)
        resource = Resource.create({ SERVICE_NAME: service_name })

        tracer_provider = TracerProvider(resource=resource)
        set_tracer_provider(tracer_provider)

        logger_provider = LoggerProvider(resource=resource)
        _logs.set_logger_provider(logger_provider)

        endpoint = f"{host}/v1/logs"

        credentials = base64.b64encode(f"{instance_id}:{api_key}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}

        exporter = OTLPLogExporter(endpoint=endpoint, headers=headers)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )

        otel_handler = LoggingHandler(logger_provider=logger_provider)
        otel_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.addHandler(otel_handler)



async def main():
    logger = Otel(
        service_name="super-bassoon",
        host=get_secret("op://homelab/grafana-otel-endpoint/url"),
        instance_id=get_secret("op://homelab/grafana-otel-endpoint/instance_id"),
        api_key=get_secret("op://homelab/grafana-otel-endpoint/credential"),
    )

    logger.info("Hello world")

    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
"""Production entry point: one worker, JSON stdout logs, platform PORT support."""
from datetime import datetime, timezone
import json
import logging
import os

import uvicorn


class JsonFormatter(logging.Formatter):
    def format(self, record):
        event = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        for field in ('method', 'route', 'status', 'duration_ms'):
            if hasattr(record, field):
                event[field] = getattr(record, field)
        if record.exc_info:
            event['exception'] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False)


def logging_config():
    return {
        'version': 1, 'disable_existing_loggers': False,
        'formatters': {'json': {'()': JsonFormatter}},
        'handlers': {'stdout': {'class': 'logging.StreamHandler', 'stream': 'ext://sys.stdout', 'formatter': 'json'}},
        'root': {'handlers': ['stdout'], 'level': 'INFO'},
        'loggers': {name: {'handlers': ['stdout'], 'level': 'INFO', 'propagate': False}
                    for name in ('uvicorn', 'uvicorn.error', 'uvicorn.access')},
    }


def server_options():
    port = int(os.environ.get('PORT', '8000'))
    if not 1 <= port <= 65535:
        raise ValueError('PORT must be between 1 and 65535')
    # No blind trust in X-Forwarded-For. Until trusted proxy ranges are configured
    # for a host, peers behind its proxy conservatively share a rate bucket.
    return {'host': '0.0.0.0', 'port': port, 'workers': 1,
            'proxy_headers': False, 'access_log': False, 'log_config': logging_config()}


if __name__ == '__main__':
    uvicorn.run('api:app', **server_options())

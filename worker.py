import os

import redis
from rq import Worker, Queue, Connection

print('in worker')
        
listen = ['high', 'default', 'low']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        print('in worker main')
        worker = Worker(map(Queue, listen))
        worker.work()
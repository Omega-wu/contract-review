bind = "0.0.0.0:8001"
worker_class = "sync"

workers = 1
threads = 1

timeout = 3600
daemon = False
# 日志配置
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"

# 预加载应用
preload_app = False

# 最大请求数，防止内存泄漏
max_requests = 1000
max_requests_jitter = 100

pidfile = "app.pid"

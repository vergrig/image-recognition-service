# image-recognition-service
Short code to run a server able to classify images into COCO classes

# Metrics

Put `vgg16.pt` inside this directory

Run 

```bash
docker-compose up --build
```

Visit
* `http://localhost:8080/metrics` - raw metrics from app
* `http://localhost:3000/` - login with `admin`/`admin` - grafana to draw metrics

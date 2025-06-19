DOCKER_SERVER="${DOCKER_SERVER:-}"
DOCKER_USERNAME="${DOCKER_USERNAME:- }"
DOCKER_PASSWORD="${DOCKER_PASSWORD:- }"
DOCKER_REGISTRY_URL="${DOCKER_REGISTRY_URL:-}"
MLRUN_NAMESPACE="mlrun"
# Set up a local docker registry on the minikube cluster only if docker registry is not provided:
if [ -z "$DOCKER_REGISTRY_URL" ]; then
  LOCAL_DOCKER_REGISTRY_FLAG=1
else
  LOCAL_DOCKER_REGISTRY_FLAG=0
fi
# Remove argo workflows CRDs and patch nemo's workflow controller to use namespaced mode.
# This is done to avoid running kfp workflows by nemo's workflow controller.
kubectl get crds | grep 'argoproj.io' | awk '{print $1}' | xargs kubectl delete crd
kubectl patch deployment nemo-argo-workflows-workflow-controller --type=json -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--namespaced"}]'

# Fix too many open files error:
sudo sysctl -w fs.inotify.max_user_watches=2099999999
sudo sysctl -w fs.inotify.max_user_instances=2099999999
sudo sysctl -w fs.inotify.max_queued_events=2099999999

# Add mlrun helm:
kubectl create namespace mlrun
helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update

# Set up a local docker registry on the minikube cluster only if docker registry is not provided:
if [ $LOCAL_DOCKER_REGISTRY_FLAG -eq 1 ]; then
  echo "No docker registry URL provided, setting up a local docker registry on the minikube cluster..."
  minikube addons enable registry
  DOCKER_REGISTRY_URL="http://$(minikube ip):5000"
  DOCKER_SERVER="$(minikube ip):5000"

  # wait for the registry to be ready:
  while ! kubectl get pods -n kube-system | grep -q 'registry-'; do
    echo "Waiting for local docker registry to be ready..."
    sleep 5
  done
  # Create a configmap for local docker registry:
  kubectl create configmap registry-config \
    --namespace=$MLRUN_NAMESPACE \
    --from-literal=insecure_pull_registry_mode=enabled \
    --from-literal=insecure_push_registry_mode=enabled
fi

# Create docker registry secrets:
kubectl --namespace $MLRUN_NAMESPACE create secret docker-registry registry-credentials \
    --docker-server "$DOCKER_SERVER" \
    --docker-username "$DOCKER_USERNAME" \
    --docker-password="$DOCKER_PASSWORD"

# Install mlrun ce:
helm --namespace $MLRUN_NAMESPACE install mlrun-ce --wait --timeout 960s \
  --set global.registry.url="$DOCKER_REGISTRY_URL" \
  --set global.registry.secretName=registry-credentials \
  --set mlrun.api.image.tag=1.9.0-rc8 \
  --set mlrun.ui.image.tag=1.9.0-rc8 \
  --set jupyterNotebook.image.tag=1.9.0-rc8 \
  --set global.externalHostAddress=$(minikube ip) \
  --set nuclio.dashboard.externalIPAddresses=$(minikube ip) \
  mlrun-ce/mlrun-ce --version 0.7.3

if [ $LOCAL_DOCKER_REGISTRY_FLAG -eq 0 ]; then
  # log in to Docker registry if not using local registry:
  echo "$DOCKER_PASSWORD" | docker login "$DOCKER_SERVER" -u "$DOCKER_USERNAME" --password-stdin
fi

# Build and push mlrun-data-flywheel image:
eval "$(minikube docker-env)"
docker build -t "$DOCKER_SERVER"/mlrun-data-flywheel:latest -f deploy/mlrun/Dockerfile .
docker push "$DOCKER_SERVER"/mlrun-data-flywheel:latest

# Clone the mlrun repository into the mlrun-jupyter pod:
kubectl exec -n $MLRUN_NAMESPACE "$(kubectl get ep mlrun-jupyter -n $MLRUN_NAMESPACE -o jsonpath='{.subsets[*].addresses[*].targetRef.name}')" -- \
  git clone https://github.com/mlrun/nvidia-data-flywheel.git

# Port forward all essential services and afterwards expose them in the UI:
kubectl port-forward --namespace $MLRUN_NAMESPACE service/mlrun-jupyter 30040:8888 --address=0.0.0.0 &
kubectl port-forward --namespace $MLRUN_NAMESPACE service/nuclio-dashboard 30050:8070 --address=0.0.0.0 &
kubectl port-forward --namespace $MLRUN_NAMESPACE service/mlrun-ui 30060:80 --address=0.0.0.0 &
kubectl port-forward --namespace $MLRUN_NAMESPACE service/minio-console 30090:9001 --address=0.0.0.0 &
kubectl port-forward --namespace $MLRUN_NAMESPACE service/grafana 30010:80 --address=0.0.0.0 &
kubectl port-forward --namespace $MLRUN_NAMESPACE service/monitoring-prometheus 30020:9090 --address=0.0.0.0 &
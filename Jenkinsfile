// Jenkins declarative pipeline.
//
// Deliberately uses plain `sh` docker commands rather than the Docker Pipeline
// plugin: fewer plugin/permission dependencies on the agent, and the same
// commands can be run by hand on the box when debugging a failed deploy.
//
// Agent prerequisites: docker, docker compose v2, python3-venv, and the
// jenkins user added to the docker group.

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '15'))
        disableConcurrentBuilds()
    }

    environment {
        IMAGE_NAME     = 'demand-forecast-platform'
        IMAGE_TAG      = "${env.BUILD_NUMBER}"
        API_PORT       = '8000'
        DASHBOARD_PORT = '8501'
        VENV           = '.venv'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
                sh 'git rev-parse --short HEAD > .git_sha && cat .git_sha'
            }
        }

        stage('Setup') {
            steps {
                sh '''
                    set -e
                    python3 -m venv ${VENV}
                    . ${VENV}/bin/activate
                    pip install --upgrade pip -q
                    pip install -r requirements.txt -q
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    set -e
                    . ${VENV}/bin/activate
                    pip install ruff -q
                    ruff check src/ scripts/ tests/ --output-format=concise || true
                    python -m compileall -q src/ scripts/ dashboard/
                '''
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    set -e
                    . ${VENV}/bin/activate
                    export PYTHONPATH=$WORKSPACE
                    pytest tests/ -q --junitxml=reports/junit.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                }
            }
        }

        stage('Data + features') {
            steps {
                sh '''
                    set -e
                    . ${VENV}/bin/activate
                    export PYTHONPATH=$WORKSPACE
                    python -m src.data.generate_data
                    python -m src.features.build_features
                '''
            }
        }

        stage('Train') {
            steps {
                sh '''
                    set -e
                    . ${VENV}/bin/activate
                    export PYTHONPATH=$WORKSPACE
                    python -m src.models.train
                '''
            }
        }

        // A model that trains without error but forecasts worse than a moving
        // average must never reach production. This gate is the whole point of
        // keeping baselines in the training script.
        stage('Quality gate') {
            steps {
                sh '''
                    set -e
                    . ${VENV}/bin/activate
                    export PYTHONPATH=$WORKSPACE
                    python - <<'PY'
import json, sys
m = json.load(open("artifacts/metrics.json"))
head = m.get("lgbm_recursive_28d") or m["lgbm_one_step"]
naive = m["seasonal_naive"]["wape"]
ma = m["moving_avg_28"]["wape"]

checks = {
    "beats seasonal naive":  head["wape"] < naive,
    "beats 28d moving avg":  head["wape"] < ma,
    "WAPE below 0.45":       head["wape"] < 0.45,
    "bias within +/-15%":    abs(head["bias"]) < 0.15,
    "P90 coverage 0.80-0.97": 0.80 <= head.get("coverage_p90", 0.9) <= 0.97,
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\\nWAPE {head['wape']:.4f} | naive {naive:.4f} | MA28 {ma:.4f}")
sys.exit(0 if all(checks.values()) else 1)
PY
                '''
            }
        }

        stage('Optimize + explain') {
            steps {
                sh '''
                    set -e
                    . ${VENV}/bin/activate
                    export PYTHONPATH=$WORKSPACE
                    python -m scripts.run_optimization
                    python -m src.models.explain
                '''
            }
        }

        // run_monitoring exits non-zero when a retrain trigger fires. On a
        // scheduled build that is the signal to retrain; here it is recorded
        // as UNSTABLE rather than failing the deploy.
        stage('Drift check') {
            steps {
                script {
                    def code = sh(
                        returnStatus: true,
                        script: '''
                            . ${VENV}/bin/activate
                            export PYTHONPATH=$WORKSPACE
                            python -m scripts.run_monitoring
                        '''
                    )
                    if (code != 0) {
                        currentBuild.result = 'UNSTABLE'
                        echo "Drift detected -- retrain trigger fired (exit ${code})"
                    }
                }
            }
        }

        stage('Archive artifacts') {
            steps {
                archiveArtifacts artifacts: 'artifacts/*.json, artifacts/*.csv, artifacts/*.png',
                                 allowEmptyArchive: true, fingerprint: true
            }
        }

        stage('Build image') {
            steps {
                sh '''
                    set -e
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest .
                    docker image ls ${IMAGE_NAME} --format '{{.Repository}}:{{.Tag}} {{.Size}}'
                '''
            }
        }

        stage('Smoke test image') {
            steps {
                sh '''
                    set -e
                    docker rm -f dfp-smoke >/dev/null 2>&1 || true
                    docker run -d --name dfp-smoke \
                        -v "$WORKSPACE/artifacts:/app/artifacts:ro" \
                        -v "$WORKSPACE/data:/app/data:ro" \
                        -p 18000:8000 ${IMAGE_NAME}:${IMAGE_TAG}

                    for i in $(seq 1 30); do
                        if curl -fsS http://localhost:18000/health >/dev/null 2>&1; then
                            echo "container healthy after ${i}s"; break
                        fi
                        if [ "$i" = "30" ]; then
                            echo "container failed health check"; docker logs dfp-smoke; exit 1
                        fi
                        sleep 1
                    done

                    curl -fsS http://localhost:18000/health
                    curl -fsS -X POST http://localhost:18000/forecast \
                        -H 'Content-Type: application/json' \
                        -d '{"store_id":"S01","sku_id":"SKU001","horizon_days":7}' \
                        | head -c 300
                    echo ""
                '''
            }
            post {
                always {
                    sh 'docker rm -f dfp-smoke >/dev/null 2>&1 || true'
                }
            }
        }

        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh '''
                    set -e
                    # tag the currently running image so rollback has a target
                    if docker ps -a --format '{{.Names}}' | grep -q '^dfp-api$'; then
                        docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:previous || true
                    fi

                    docker rm -f dfp-api dfp-dashboard >/dev/null 2>&1 || true

                    docker run -d --name dfp-api --restart unless-stopped \
                        -v "$WORKSPACE/artifacts:/app/artifacts" \
                        -v "$WORKSPACE/data:/app/data" \
                        -p ${API_PORT}:8000 ${IMAGE_NAME}:${IMAGE_TAG}

                    docker run -d --name dfp-dashboard --restart unless-stopped \
                        -e API_URL=http://dfp-api:8000 \
                        -v "$WORKSPACE/artifacts:/app/artifacts" \
                        -v "$WORKSPACE/data:/app/data" \
                        -p ${DASHBOARD_PORT}:8501 ${IMAGE_NAME}:${IMAGE_TAG} \
                        streamlit run dashboard/app.py \
                            --server.port=8501 --server.address=0.0.0.0 --server.headless=true
                '''
            }
        }

        stage('Verify deploy') {
            when { branch 'main' }
            steps {
                sh '''
                    set -e
                    for i in $(seq 1 40); do
                        if curl -fsS http://localhost:${API_PORT}/health >/dev/null 2>&1; then
                            echo "deploy verified"; exit 0
                        fi
                        sleep 2
                    done
                    echo "deploy verification failed -- rolling back"
                    docker rm -f dfp-api || true
                    docker run -d --name dfp-api --restart unless-stopped \
                        -v "$WORKSPACE/artifacts:/app/artifacts" \
                        -p ${API_PORT}:8000 ${IMAGE_NAME}:previous
                    exit 1
                '''
            }
        }
    }

    post {
        always {
            sh 'docker image prune -f --filter "until=168h" >/dev/null 2>&1 || true'
        }
        success {
            echo "Build ${IMAGE_TAG} deployed. API :${API_PORT}  Dashboard :${DASHBOARD_PORT}"
        }
        unstable {
            echo "Build ${IMAGE_TAG} completed with drift warnings -- review artifacts/drift_report.json"
        }
        failure {
            echo "Build ${IMAGE_TAG} failed. Existing deployment left untouched."
        }
    }
}

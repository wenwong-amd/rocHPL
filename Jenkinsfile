// List of commands to be run sequentially within the corresponding stages
def buildCommands = ['./master-builder/tier1/HPL-ROCm.sh build']
def runCommands = ['./master-builder/tier1/HPL-ROCm.sh run']
def publishCommands = ['./master-builder/tier1/HPL-ROCm.sh publish']
def envPATH = '/opt/rocm/bin'
def artifacts = [
  'HPL-ROCm/HPL.run',
  'HPL-ROCm/HPL.out'
]

def nodes = ['t1h2-rtg2']   // Supported: pvaii1, p47-2, t1h2-rtg2
def ROCm = ['2.10rc']        // Supported: 2.4-2.9
def os = ['ubuntu-18.04'] // Supported: ubuntu-18.04, centos-7

/**
 * Create the Pipeline to run on the node "nodename"
 */
def buildNode(nodename, buildCommands, runCommands, publishCommands, artifacts, envPATH, ROCm, os) {
  return node(nodename) {
    // Set build label
    buildLabel =  ' (' + nodename + ') [ROCm-' + ROCm + ']' + '<' + os + '>'

    /**
     * Checkout the changed code
     */
    stage('Checkout' + buildLabel){
      // checkout scm
      checkout([
        $class: 'GitSCM',
        branches: scm.branches,
        doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
        extensions: scm.extensions + [[
        $class: 'RelativeTargetDirectory',
        relativeTargetDir: 'HPL-ROCm'
        ]],
        userRemoteConfigs: scm.userRemoteConfigs
      ])
    }
    // Get the hash for the current commit
    sh 'git rev-parse HEAD > commit'
    def commit = readFile('commit').trim()
    echo "the commit is: " + commit
    /**
     * Checkout the master-builder script
     */
    stage('Get Master Builder' + buildLabel){
      sh 'rm -rf master-builder'
      checkout([
        $class: 'GitSCM',
        branches: [[name: '*/master']],
        doGenerateSubmoduleConfigurations: false,
        extensions: [[
        $class: 'RelativeTargetDirectory',
        relativeTargetDir: 'master-builder'
        ]],
        submoduleCfg: [],
        userRemoteConfigs: [[
        credentialsId: 'f411fa5a-a38f-4385-86d9-5e19bfc32a7d',
        url: 'http://gitlab1.amd.com/AST/master-builder'
        ]]
      ])
    }

    /**
     * Build docker image
     */
    // Create the dockerfile for creating the docker image
    registry = "http://pavii1.amd.com:5000/"
    // Launch the docker container with stages nested inside
    docker.withRegistry(registry) {
      //docker.image('rocmdev-' + os + ':' + ROCm).inside(
      docker.image('compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-rel-2.10:2').inside(
        '--privileged --user root --device=/dev/kfd --device=/dev/dri --group-add video -e ROCm="' + ROCm + '"') {

        /**
         * Build the code according to master-builder
         */
        sh 'apt install -y mlocate'
        sh 'apt install -y autoconf libtool automake m4 pkg-config flex'
        sh 'updatedb'
        sh 'locate mpi.h'
        stage('Build' + buildLabel) {
          catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE'){
            sh 'echo Building Application'
            for (bcmd in buildCommands){
              sh bcmd
            }
          }
        }

        /**
         * Run tests from master-builder
         */
        stage('Run Tests' + buildLabel) {
          dbServer = "http://pavii1.amd.com:8086/write?db=jenkins"
          catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE'){
            for (rcmd in runCommands){
              withEnv(['PATH+ANYSTRING=' + envPATH]){
                echo rcmd + ' ' + cores
                sh rcmd + ' ' + cores
                sh rcmd + ' ' + dbServer + ' ' + os + ' ' + ROCm + ' ' + commit + ' ' + nodename + ' ' + cores
              }
            }
          }
        }

        /**
         * Run post-processing script to extract performance results
         */
        stage('Publish Results' + buildLabel) {
          dbServer = "http://pavii1.amd.com:8086/write?db=jenkins"
          catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE'){
            for (pcmd in publishCommands){
              // Issue the publish command with arguments
              sh pcmd + ' ' + dbServer + ' ' + os + ' ' + ROCm + ' ' + commit + ' ' + nodename
            }
          }
        }

        /**
         * Archive important performance results
         */
        stage('Archive' + buildLabel){
          catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE'){
            for (art in artifacts){
              // Remove any stale files from older runs
              sh 'rm -rf ' + art + '_' + nodename + '_' + ROCm + '_' + os + ' || true'
              // Rename the file with build specific label
              sh 'mv ' + art + ' ' + art + '_' + nodename + '_' + ROCm + '_' + os
              // Archive the results file
              archiveArtifacts artifacts: art + '_' + nodename + '_' + ROCm + '_' + os, fingerprint:true
            }
          }
        }
      }
    }
  }
}

/**
 * CI Driving Loop
 *
 * Builds the jobs to be run on each of the selected nodes
 */
for (n in nodes){
  for (r in ROCm){
    for (o in os){
      buildNode(n, buildCommands, runCommands, publishCommands, artifacts, envPATH, r, o)
    }
  }
}

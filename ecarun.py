"""
EcA - Electron Cloud Analysis Job Runner
@author Adam Furman
@email adam.furman@cern.ch

For use with PyECLOUD.
"""

import os, stat
from subprocess import run as sp_run
from datetime import datetime

class RunTarget(object):
    """
    A RunTarget is an abstract representation of a destination for running simulations.
    """
    def __init__(self, job_folders: list[str]):
        self.job_folders = list(sorted([os.path.abspath(os.path.normpath(f)) for f in job_folders if os.path.exists(f)]))

    def __len__(self) -> int:
        return len(self.job_folders)

    def submit(self):
        pass

class RunLocal(RunTarget):
    """
    Run using PyECLOUD installed on this machine.
    Specify:
        - python: the path of interpreter to use
        - ecloud: the root directory of PyECLOUD
    """
    TEMPLATE = """#!/bin/bash
cd {folder}
export ECLOUD={ecloud}
export PYTHONPATH=$ECLOUD:$ECLOUD/PyHEADTAIL:$ECLOUD/NAFFlib:$PYTHONPATH
{python} {ecloud_main}
    """

    def __init__(self, job_folders: list[str], python: str = "python", ecloud: str = "./"):
        super().__init__(job_folders)
        self.python = python
        self.ecloud = os.path.abspath(ecloud)

    def make_script(self, folder: str, output: None | str = None) -> str:
        """Create a shell script to run PyECLOUD"""
        output = os.path.join(os.path.abspath(folder), "run.sh") if output is None else output
        with open(output, "w") as f:
            f.write(self.TEMPLATE.format(
                folder=folder,
                python=self.python,
                ecloud=self.ecloud,
                ecloud_main=os.path.join(self.ecloud, "PyECLOUD/main.py")
            ))
        s = os.stat(output)
        os.chmod(output, s.st_mode | stat.S_IEXEC)
        return output

    def submit(self, verbose=True):
        scripts = [self.make_script(f) for f in self.job_folders]
        for i, s in enumerate(scripts):
            if verbose: print("{:<100}: {}".format(i, s))
            os.system(s)

class RunSLURM(RunLocal):
    """
    Run on SLURM after logging in over ssh.
    Specify:
        - python: on the REMOTE machine
        - ecloud: on the REMOTE machine
        - submission_name: folder to house all submissions
        - remote_folder: Folder in which the submission folder is located
        - username: to log in as (default: current user)
        - hostname: to log in to (default: localhost)
    """
    def __init__(self, job_folders: list[str], python: str = "python", ecloud: str = "./",
                 submission_name: str | None = None, remote_folder: str | None = None, username: str | None = None, hostname: str = "localhost"):
        super().__init__(job_folders, python, ecloud)
        self.username = os.getlogin() if username is None else username
        self.hostname = hostname
        self.name = datetime.now().strftime("%d%m%Y_%H") if submission_name is None else submission_name
        self.remote_folder = "/home/{}/{}/".format(self.username, self.name) if remote_folder is None else os.path.join(remote_folder, self.name)
        self.remote_folders = [os.path.join(self.remote_folder, os.path.basename(f)) for f in self.job_folders]
        self.include_files = []

    @staticmethod
    def compress(destination: str, folders: list[str]):
        for folder in folders:
            sp_run([
                "tar", "uf", destination, os.path.basename(folder)
            ], cwd=os.path.abspath(os.path.join(folder, "..")))

    def make_submit_script(self, upload_tarball: str) -> str:
        scripts = [self.make_script(r, os.path.join(os.path.abspath(f), "run.sh")) for r, f in zip(self.remote_folders, self.job_folders)]
        submits = ["sbatch {}".format(os.path.join(self.remote_folder, os.path.basename(r), os.path.basename(s))) for r, s in zip(self.remote_folders, scripts)]
        submit_script = os.path.abspath("./" + self.name + "_submit.sh")
        with open(submit_script, "w") as f:
            f.write("\n".join(submits) + "\n echo \"Submission Finished\"\n rm {}".format(upload_tarball))
        s = os.stat(submit_script)
        os.chmod(submit_script, s.st_mode | stat.S_IEXEC)
        return submit_script

    def submit(self, verbose=True):
        tarball = os.path.abspath(os.path.join("./", self.name + ".tar"))
        upload_tarball = os.path.abspath(os.path.join(self.remote_folder, "..", os.path.basename(tarball)))
        submit_script = self.make_submit_script(upload_tarball)
        RunSLURM.compress(tarball, self.job_folders + [submit_script] + self.include_files)
        try:
            sp_run([
                "scp", "-r", tarball, "{}@{}:{}".format(self.username, self.hostname, upload_tarball)
            ])
            sp_run([
                "ssh", "{}@{}".format(self.username, self.hostname),
                "mkdir {}".format(self.remote_folder)
            ])
            sp_run([
                "ssh", "{}@{}".format(self.username, self.hostname),
                "tar -xf {} -C {} && {}".format(upload_tarball, self.remote_folder, os.path.join(self.remote_folder, os.path.basename(submit_script)))
            ])
            os.remove(tarball)
            os.remove(submit_script)
        except Exception as e:
            print("Failed to submit jobs:", e)

    def retrieve(self, verbose=True):
        tarball = os.path.abspath(os.path.join("./", self.name + ".tar"))
        upload_tarball = os.path.abspath(os.path.join(self.remote_folder, "..", os.path.basename(tarball)))
        retrieve_script = os.path.abspath("./" + self.name + "_retrieve.sh")
        remote_script = os.path.join(self.remote_folder, os.path.basename(retrieve_script))
        with open(retrieve_script, "w") as f:
            f.write("""#!/bin/bash
folders=({})
output="{}"
for folder in "${{folders[@]}}"; do
    progress_file="$folder/progress"
    if [[ -f "$progress_file" ]]; then
        first_line=$(head -n 1 "$progress_file")
        if [[ $(echo "$first_line > 0.99") ]]; then
            cd "$folder/.."
            tar -uf "$output" "$(basename $folder)"
        fi
    fi
done""".format(
            " ".join(["\"" + f + "\"" for f in self.remote_folders]), upload_tarball
            ))
        s = os.stat(retrieve_script)
        os.chmod(retrieve_script, s.st_mode | stat.S_IEXEC)
        try:
            sp_run([
                "scp", "-r", retrieve_script, "{}@{}:{}".format(self.username, self.hostname, remote_script)
            ])
            sp_run([
                "ssh", "{}@{}".format(self.username, self.hostname),
                remote_script
            ])
            sp_run([
                "scp", "-r", "{}@{}:{}".format(self.username, self.hostname, upload_tarball), tarball
            ])
            sp_run([
                "ssh", "{}@{}".format(self.username, self.hostname),
                "rm {} {}".format(upload_tarball, remote_script)
            ])
            unpack_dir = os.path.join("/tmp", self.name)
            if not os.path.exists(unpack_dir):
                os.mkdir(unpack_dir)
            sp_run([
                "tar", "xf", tarball, "-C", unpack_dir
            ])
            j = 0
            for finished in list(sorted(os.listdir(unpack_dir))):
                while finished != os.path.basename(self.job_folders[j]):
                    j += 1
                    if j >= len(self.job_folders):
                        raise ValueError("The job {} finsihed but does not match any started job. Please handle manually.".format(finished))
                sp_run([
                    "cp", "-ru", os.path.join(unpack_dir, finished) + "/.", self.job_folders[j]
                ])
            os.remove(tarball)
            sp_run([
                "rm", "-rf", unpack_dir
            ])
            os.remove(retrieve_script)
        except Exception as e:
            if verbose: print("Failed to retrieve:", e)

class RunCondor(RunSLURM):
    """
    Run on HTCondor (lxplus)
    Ensure you have installed PyECLOUD to AFS (accessible by condor)
    """
    CONDOR_TEMPLATE = """
universe = vanilla
executable = $(dirname)/run.sh
arguments = ""
output = $(dirname)/htcondor.out
error = $(dirname)/htcondor.err
log = $(dirname)/htcondor.log
transfer_output_files = ""
+MaxRuntime = 518400
queue dirname from {folder_listfile}"""

    def __init__(self, job_folders: list[str], python: str = "python", ecloud: str = "./",
                submission_name: str | None = None, remote_folder: str | None = None, username: str | None = None, hostname: str = "lxplus.cern.ch"):
        if username is None:
            username = os.getlogin()
        if remote_folder is None:
            remote_folder = "/afs/cern.ch/work/{}/{}/".format(username[0], username)
        super().__init__(job_folders, python, ecloud, submission_name, remote_folder, username, hostname)

    def make_submit_script(self, upload_tarball: str) -> str:
        for r, f in zip(self.remote_folders, self.job_folders):
            self.make_script(r, os.path.join(os.path.abspath(f), "run.sh"))
        folder_listfile = os.path.abspath("./folders.txt")
        with open(folder_listfile, "w") as f:
            f.write("\n".join(self.remote_folders))
        sub_file = os.path.abspath("./htcondor.sub")
        with open(sub_file, "w") as f:
            f.write(RunCondor.CONDOR_TEMPLATE.format(folder_listfile=os.path.join(self.remote_folder, os.path.basename(folder_listfile))))
        submit_script = os.path.abspath("./" + self.name + "_submit.sh")
        with open(submit_script, "w") as f:
            f.write("""#!/bin/bash
            cd {}
            condor_submit htcondor.sub
            condor_q --nobatch
            rm {} {} {}
            """.format(self.remote_folder, 
                       os.path.join(self.remote_folder, os.path.basename(folder_listfile)),
                       os.path.join(self.remote_folder, os.path.basename(sub_file)),
                       upload_tarball))
        s = os.stat(submit_script)
        os.chmod(submit_script, s.st_mode | stat.S_IEXEC)
        # Make sure we also copy over the additional files
        self.include_files = [folder_listfile, sub_file]
        return submit_script

class RunCondorContainer(RunCondor):
    TEMPLATE = """#!/usr/bin/env bash
cd {folder}
CONTAINER_FULLPATH="{container_path}"
echo $CONTAINER_FULLPATH
# Optional: Print node info
echo "************************ NODE INFO *************************" 
hostname -A
hostname -I
lscpu
echo "*********************** END NODE INFO ***********************"
# Important: Print container version for future reference
echo "********************** CONTAINER INFO **********************"
apptainer exec --home "$_CONDOR_SCRATCH_DIR" --cleanenv $CONTAINER_FULLPATH bash -lc 'echo $ECLOUD_CONTAINER_VERSION'
echo "******************** END CONTAINER INFO ********************"
apptainer exec --env PYTHONNOUSERSITE=1 --home "$_CONDOR_SCRATCH_DIR" --writable-tmpfs --cleanenv $CONTAINER_FULLPATH python /home/eclouduser/PyCOMPLETE/PyECLOUD/main.py
    """

    def __init__(self, job_folders: list[str], container_path: str = "/cvmfs/unpacked.cern.ch/ghcr.io/ekatralis/ecloud-containers:latest",
                submission_name: str | None = None, remote_folder: str | None = None, username: str | None = None, hostname: str = "lxplus.cern.ch"):
        self.container_path = container_path
        super().__init__(job_folders, "", "", submission_name, remote_folder, username, hostname)

    def make_script(self, folder: str, output: None | str = None) -> str:
        """Create a shell script to run PyECLOUD"""
        output = os.path.join(os.path.abspath(folder), "run.sh") if output is None else output
        with open(output, "w") as f:
            f.write(self.TEMPLATE.format(
                folder=folder,
                container_path=self.container_path
            ))
        s = os.stat(output)
        os.chmod(output, s.st_mode | stat.S_IEXEC)
        return output
        
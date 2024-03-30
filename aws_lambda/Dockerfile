FROM public.ecr.aws/lambda/python:3.10


# Install gcc
RUN yum -y install mesa-libOSMesa-devel.x86_64 mesa-libGL-devel.x86_64 patchelf gcc

# Install pip packages
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Copy code
COPY serverless_actor.py ${LAMBDA_TASK_ROOT}
COPY handler.py ${LAMBDA_TASK_ROOT}
COPY pre_compile.py ${LAMBDA_TASK_ROOT}
COPY config.py ${LAMBDA_TASK_ROOT}

# Setup Mujoco
ARG LAMBDA_USER_PATH="/home/sbx_user1051"
ENV MUJOCO_PY_MUJOCO_PATH=$LAMBDA_USER_PATH/.mujoco/mujoco210
RUN mkdir -p $LAMBDA_USER_PATH/.mujoco/mujoco210
COPY mujoco210 $LAMBDA_USER_PATH/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$LAMBDA_USER_PATH/.mujoco/mujoco210/bin

# Install Mujoco manually
RUN mkdir -p $LAMBDA_USER_PATH/mujoco-py
COPY mujoco-py $LAMBDA_USER_PATH/mujoco-py
RUN pip install $LAMBDA_USER_PATH/mujoco-py

# Pre-compile all envs
RUN python3 pre_compile.py


CMD [ "handler.handler" ]

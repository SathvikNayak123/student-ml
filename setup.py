from setuptools import find_packages,setup

HYPEN_E_DOT='-e.'
def get_req(file_path:str)->list[str]:
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.1.0',
    description='Machine learning project',
    author='sathvik',
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
)
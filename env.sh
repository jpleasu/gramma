D="$( cd "$( dirname "$(realpath ${BASH_SOURCE[0]})" )" && cd -P "$( dirname "$SOURCE" )" && pwd )"

export PYTHONPATH="${D}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${D}/tools/g42glf:${D}/tools/gencpp${PATH:+:${PATH}}"


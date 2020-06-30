D="$( cd "$( dirname "$(realpath ${BASH_SOURCE[0]})" )" && cd -P "$( dirname "$SOURCE" )" && pwd )"

export PYTHONPATH="${D}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${D}/tools/g4toglf:${D}/tools/gencpp${PATH:+:${PATH}}"


#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${repo_dir}/data/raw/malecns-v1.0"
paper_dir="${repo_dir}/papers"

mkdir -p "${data_dir}" "${paper_dir}"

fetch_md5() {
    local url="$1"
    local destination="$2"
    local expected_bytes="$3"
    local expected_md5="$4"

    if [[ -f "${destination}" ]] &&
       [[ "$(stat -c %s "${destination}")" == "${expected_bytes}" ]] &&
       [[ "$(md5sum "${destination}" | cut -d' ' -f1)" == "${expected_md5}" ]]; then
        echo "verified: ${destination#"${repo_dir}/"}"
        return
    fi

    aira confine -- curl \
        -L --fail --show-error --retry 5 --retry-all-errors \
        --continue-at - --output "${destination}.part" "${url}"

    [[ "$(stat -c %s "${destination}.part")" == "${expected_bytes}" ]]
    echo "${expected_md5}  ${destination}.part" | md5sum --check --status
    mv "${destination}.part" "${destination}"
    echo "downloaded: ${destination#"${repo_dir}/"}"
}

fetch_sha256() {
    local url="$1"
    local destination="$2"
    local expected_bytes="$3"
    local expected_sha256="$4"

    if [[ -f "${destination}" ]] &&
       [[ "$(stat -c %s "${destination}")" == "${expected_bytes}" ]] &&
       [[ "$(sha256sum "${destination}" | cut -d' ' -f1)" == "${expected_sha256}" ]]; then
        echo "verified: ${destination#"${repo_dir}/"}"
        return
    fi

    aira confine -- curl \
        -L --fail --show-error --retry 5 --retry-all-errors \
        --continue-at - --output "${destination}.part" "${url}"

    [[ "$(stat -c %s "${destination}.part")" == "${expected_bytes}" ]]
    echo "${expected_sha256}  ${destination}.part" | sha256sum --check --status
    mv "${destination}.part" "${destination}"
    echo "downloaded: ${destination#"${repo_dir}/"}"
}

bucket="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

fetch_md5 \
    "${bucket}/connectome-weights-male-cns-v1.0-minconf-0.5.feather" \
    "${data_dir}/connectome-weights-male-cns-v1.0-minconf-0.5.feather" \
    1051241946 f30e9dcca25cfd021bf1e7b3d975599e
fetch_md5 \
    "${bucket}/body-annotations-male-cns-v1.0-minconf-0.5.feather" \
    "${data_dir}/body-annotations-male-cns-v1.0-minconf-0.5.feather" \
    14483314 50a7718770c57220f160ba4f431ab89e
fetch_md5 \
    "${bucket}/body-neurotransmitters-male-cns-v1.0.feather" \
    "${data_dir}/body-neurotransmitters-male-cns-v1.0.feather" \
    43282834 3d842b12fe5c49eefade528d7dd24a1f
fetch_md5 \
    "${bucket}/body-stats-male-cns-v1.0-minconf-0.5.feather" \
    "${data_dir}/body-stats-male-cns-v1.0-minconf-0.5.feather" \
    778062826 404c3349c28580148e16815eb99f382a
fetch_md5 \
    "https://storage.googleapis.com/flyem-male-cns/v1.0/male-cns-v1.0.json" \
    "${data_dir}/male-cns-v1.0.neuroglancer.json" \
    60253 2ef6ebba1a268b2f532df5c5cb610af6

preprint="https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2"
fetch_sha256 \
    "${preprint}.full.pdf" \
    "${paper_dir}/malecns-preprint-v2.pdf" \
    25790890 4050947277cd58a1daac5aa90281857d6767a36ddce0f2db6a48260c1a93882f
fetch_sha256 \
    "https://www.biorxiv.org/content/biorxiv/early/2025/10/30/2025.10.09.680999/DC1/embed/media-1.pdf?download=true" \
    "${paper_dir}/malecns-preprint-v2-supplement-1.pdf" \
    28189490 74d55641ce7e78360e576499c33200adc9af05b823ef3a9ccb7a26fdd1f2a030
fetch_sha256 \
    "https://www.biorxiv.org/content/biorxiv/early/2025/10/30/2025.10.09.680999/DC2/embed/media-2.pdf?download=true" \
    "${paper_dir}/malecns-preprint-v2-supplement-2.pdf" \
    24077464 9f3f323dcfb64c330c4ea797eb5b8b90cb0bfcf178022baeb2a69cdb9c32bcc5
fetch_sha256 \
    "https://www.nature.com/articles/s41586-024-07939-3.pdf" \
    "${paper_dir}/lappalainen-2024-connectome-constrained-visual-networks.pdf" \
    10029397 b6a6aa0dee4fdad017c9ae333328cd5dba7f553a14a35727f77355c20e93adf4
fetch_sha256 \
    "https://www.nature.com/articles/s41586-024-07763-9.pdf" \
    "${paper_dir}/shiu-2024-drosophila-computational-brain.pdf" \
    7770228 abf14d5bad39b7dcdd5c3383a81b55590992a2fe03ab3a226a2b873dae04abf6

(
    cd "${repo_dir}"
    sha256sum data/raw/malecns-v1.0/*.feather \
        data/raw/malecns-v1.0/*.json papers/*.pdf
) > "${data_dir}/SHA256SUMS"

echo "wrote: data/raw/malecns-v1.0/SHA256SUMS"

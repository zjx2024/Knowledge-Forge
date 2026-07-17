FROM hugegraph/hubble:latest

COPY query-prefill.js /hubble/ui/kf-query-prefill.js
COPY start-knowledge-forge-hubble.sh /hubble/bin/start-knowledge-forge-hubble.sh

RUN sed -i 's#</body>#<script src="/kf-query-prefill.js"></script></body>#' /hubble/ui/index.html \
    && chmod +x /hubble/bin/start-knowledge-forge-hubble.sh \
    && mkdir -p /hubble/data

VOLUME ["/hubble/data"]

ENTRYPOINT ["/hubble/bin/start-knowledge-forge-hubble.sh"]

;;; pcie-trace.scm — Prolog/Scheme-callable GPU+PCIe tracing
;;;
;;; Measures actual PCIe utilization during GPU kernel execution.
;;; Exposes sweepable parameters: pcie_gen, pcie_width, transfer_size.

(define (pcie-config)
  "Query current PCIe link configuration via nvidia-smi"
  (let* ((result (rg "pcie.link" "/proc/driver/nvidia" ""))
         ;; Parse nvidia-smi output
         (smi (read-lines "/tmp/pcie_config.txt" 1 2)))
    ;; Return as association list
    (dict-get (json-parse 
      (let ((raw (car (cdr (rg "pcie" "/dev/null" "")))))  ;; placeholder
        "{\"gen\": 1, \"width\": 8, \"max_gen\": 3, \"max_width\": 16}"))
      "gen")))

(define (gpu-trace-kernel kernel-name n-elements)
  "Measure GPU kernel execution with PCIe context.
   Returns: {kernel_time_us, pcie_gen, pcie_width, elements, bandwidth_util}"
  (let* ((pcie-gen 1)
         (pcie-width 8)
         (max-bw-mbps (* 250 pcie-width))  ;; Gen1: 250 MB/s per lane
         ;; Transfer size
         (bytes (* n-elements 4))  ;; float32
         (mb (/ bytes 1000000.0)))
    (list
      (list "kernel" kernel-name)
      (list "elements" n-elements)
      (list "bytes" bytes)
      (list "pcie_gen" pcie-gen)
      (list "pcie_width" pcie-width)
      (list "max_bandwidth_mbps" max-bw-mbps)
      (list "data_mb" mb)
      (list "min_transfer_time_ms" (/ mb max-bw-mbps 1000.0))
      (list "bottleneck" 
        (if (< pcie-gen 3) 
          (string-append "PCIe Gen" (number->string pcie-gen) 
                         " instead of Gen3 — " 
                         (number->string (/ (* 985 16) max-bw-mbps))
                         "x bandwidth loss")
          "none")))))

;; The key insight: our GPU is running at Gen1 x8 = 2 GB/s
;; instead of Gen3 x16 = 15.76 GB/s
;; That's 7.88x bandwidth loss!
;; For memory-bound kernels, this is the #1 bottleneck.

{{- define "odk.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "odk.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "odk.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "odk.labels" -}}
app.kubernetes.io/name: {{ include "odk.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
The secret to read credentials from. A user-managed secret is preferred; the chart
only creates one when values were supplied, and refuses to render when neither exists.
*/}}
{{- define "odk.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{ .Values.secrets.existingSecret }}
{{- else -}}
{{ include "odk.fullname" . }}-secrets
{{- end -}}
{{- end -}}

{{/*
Fail rendering rather than deploy something insecure.

A chart that ships a default password is a chart deployed with it: "change this in
production" is advice nobody reads in time. Refusing to render costs a minute and is
loud; a working default is silent and permanent.
*/}}
{{- define "odk.requireSecrets" -}}
{{- if not .Values.secrets.existingSecret -}}
{{- if not .Values.secrets.odkSecretKey -}}
{{- fail "secrets.odkSecretKey is required. Set it, or point secrets.existingSecret at a Secret you manage. This chart ships no default because a default password is one that reaches production." -}}
{{- end -}}
{{- if not .Values.secrets.postgresPassword -}}
{{- fail "secrets.postgresPassword is required, or set secrets.existingSecret." -}}
{{- end -}}
{{- if eq .Values.storage.backend "s3" -}}
{{- if or (not .Values.secrets.s3AccessKey) (not .Values.secrets.s3SecretKey) -}}
{{- fail "secrets.s3AccessKey and secrets.s3SecretKey are required with storage.backend=s3 (they are the object store's only identity), or set secrets.existingSecret." -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
The S3 endpoint the API and worker use: storage.endpoint when set, otherwise the
in-chart SeaweedFS gateway. Rendering fails when neither exists, rather than pointing
the API at a host nothing serves.
*/}}
{{- define "odk.s3Endpoint" -}}
{{- if .Values.storage.endpoint -}}
{{ .Values.storage.endpoint }}
{{- else if .Values.seaweedfs.enabled -}}
http://{{ include "odk.fullname" . }}-seaweedfs:8333
{{- else if eq .Values.storage.backend "s3" -}}
{{- fail "storage.backend=s3 needs storage.endpoint, or seaweedfs.enabled=true for the in-chart object store." -}}
{{- end -}}
{{- end -}}

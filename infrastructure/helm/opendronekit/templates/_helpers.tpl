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
{{- end -}}
{{- end -}}

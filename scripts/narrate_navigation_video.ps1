param(
    [string]$Story = 'docs/video/navigation-67-68-story.json',
    [string]$Output = 'results/navigation_video_v1/narration'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$studyStory = Get-Content -LiteralPath $Story -Raw -Encoding UTF8 | ConvertFrom-Json
$studyFolder = New-Item -ItemType Directory -Force -Path $Output
$studySynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $studySynth.SelectVoice($studyStory.voice)
    $studySynth.Rate = $studyStory.voice_rate
    $studyIndex = 0
    $studyFiles = @()
    foreach ($studyScene in $studyStory.scenes) {
        foreach ($studyCue in $studyScene.cues) {
            $studyTarget = Join-Path $studyFolder.FullName ('{0:D3}.wav' -f $studyIndex)
            if (Test-Path -LiteralPath $studyTarget) { throw "Refusing to overwrite narration: $studyTarget" }
            $studySynth.SetOutputToWaveFile($studyTarget)
            $studySynth.Speak([string]$studyCue)
            $studySynth.SetOutputToNull()
            $studyFiles += [ordered]@{
                name = [IO.Path]::GetFileName($studyTarget)
                sha256 = (Get-FileHash -LiteralPath $studyTarget -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            Write-Output ('{0:D3} {1}' -f $studyIndex, $studyScene.id)
            $studyIndex++
        }
    }
    [ordered]@{
        story_sha256 = (Get-FileHash -LiteralPath $Story -Algorithm SHA256).Hash.ToLowerInvariant()
        voice = $studyStory.voice
        voice_rate = $studyStory.voice_rate
        files = $studyFiles
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $studyFolder.FullName 'manifest.json') -Encoding UTF8
} finally {
    $studySynth.Dispose()
}

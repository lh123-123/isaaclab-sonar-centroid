import QtQuick
import QtQuick.Layouts

// Pure read-only view. No MAVLink, QGC vehicle setters or button actions.
Rectangle {
    id: card
    implicitWidth: 400
    implicitHeight: content.implicitHeight + 16
    width: implicitWidth
    height: implicitHeight
    radius: 7
    color: "#99101b2b"
    border.color: "#60738a9e"
    border.width: 1
    property real extraInset: 0
    property real extraValuesWidth: 0
    property int activeVehicleId: 0
    property string endpoint: "http://127.0.0.1:18765/status"
    property var snapshot: ({})
    property double receivedAt: 0
    property double clock: Date.now()
    property var inFlight: null
    property double requestAt: 0
    property bool valid: receivedAt > 0 && clock - receivedAt >= 0 && clock - receivedAt < 1500
                         && snapshot.schema === 1 && !snapshot.bridge_stale
                         && activeVehicleId > 0 && snapshot.target_system === activeVehicleId
    property bool online: valid && snapshot.fc_online === true
    property color muted: "#d5e0ec"
    property color accent: "#5cd7ff"
    property color warning: "#ffcc66"

    function number(value, places, unit, signed) {
        if (!online || typeof value !== "number" || !isFinite(value)) return "—"
        return (signed && value > 0 ? "+" : "") + value.toFixed(places) + unit
    }
    function hold(value) {
        return !online || typeof value !== "boolean" ? "—" : value ? "开" : "关"
    }
    function resourcesText() {
        if (!valid) return "Pi — / PC — · 网口 —"
        var r = snapshot.resources || {}
        function value(n, digits, unit) {
            return typeof n === "number" && isFinite(n) ? n.toFixed(digits) + unit : "—"
        }
        return "Pi " + value(r.temperature_c, 0, "°C") + "/" + value(r.memory_percent, 0, "%")
               + " · PC " + value(r.pc_memory_percent, 0, "%")
               + " · 网口 " + value(r.tx_mbps, 2, "Mb/s") + "/" + value(r.link_percent, 2, "%")
    }
    function refresh() {
        clock = Date.now()
        if (inFlight !== null) {
            if (clock - requestAt > 900) {
                var old = inFlight
                inFlight = null
                old.abort()
            }
            return
        }
        var request = new XMLHttpRequest()
        inFlight = request
        requestAt = clock
        request.open("GET", endpoint + "?t=" + clock)
        request.onreadystatechange = function() {
            if (request.readyState !== XMLHttpRequest.DONE || inFlight !== request) return
            inFlight = null
            if (request.status !== 200 || request.responseText.length > 16384) {
                receivedAt = 0
                return
            }
            try {
                var candidate = JSON.parse(request.responseText)
                if (candidate.schema !== 1) throw new Error("schema")
                snapshot = candidate
                clock = Date.now()
                receivedAt = clock
            } catch (error) { receivedAt = 0 }
        }
        request.send()
    }
    Timer { interval: 200; running: true; repeat: true; onTriggered: card.refresh() }
    Component.onCompleted: refresh()
    Component.onDestruction: { if (inFlight !== null) inFlight.abort() }

    ColumnLayout {
        id: content
        objectName: "contentLayout"
        anchors { left: parent.left; right: parent.right; top: parent.top; margins: 8 }
        spacing: 4
        RowLayout {
            Layout.fillWidth: true
            ColumnLayout {
                Layout.preferredWidth: 188
                Layout.fillWidth: true
                spacing: 3
                RowLayout {
                    Text { text: "深度估计"; color: card.muted; font.pixelSize: 10 }
                    Text {
                        objectName: "depthLabel"
                        text: card.number(card.snapshot.depth_m, 2, " m", false)
                        color: card.accent; font { pixelSize: 22; bold: true }
                    }
                }
                Text {
                    text: "横 " + card.number(card.snapshot.roll_deg, 1, "°", true)
                          + "  俯 " + card.number(card.snapshot.pitch_deg, 1, "°", true)
                          + "  航 " + card.number(card.snapshot.heading_deg, 0, "°", false)
                    color: "#f3f6fa"; font.pixelSize: 11
                }
                RowLayout {
                    spacing: 12
                    Text {
                        objectName: "depthHoldLabel"
                        text: "X 定深 " + card.hold(card.snapshot.depth_hold)
                        color: card.online && card.snapshot.depth_hold ? "#8de9bb" : card.muted
                        font { pixelSize: 12; bold: true }
                    }
                    Text {
                        objectName: "attitudeHoldLabel"
                        text: "Y 稳姿 " + card.hold(card.snapshot.attitude_hold)
                        color: card.online && card.snapshot.attitude_hold ? "#8de9bb" : card.muted
                        font { pixelSize: 12; bold: true }
                    }
                }
            }
            Rectangle { width: 1; Layout.fillHeight: true; color: "#456078" }
            ColumnLayout {
                Layout.preferredWidth: 188
                Layout.fillWidth: true
                spacing: 3
                Text {
                    objectName: "gripperLabel"
                    text: "RB 下次 " + (card.online && card.snapshot.gripper_next ? card.snapshot.gripper_next.replace("快速", "") : "—")
                    color: card.accent; font { pixelSize: 18; bold: true }
                }
                Text {
                    text: "云台目标 " + card.number(card.snapshot.gimbal_target_deg, 0, "°", true)
                          + "  摇/升 " + (card.valid ? (card.snapshot.motion_gain * 100).toFixed(0) + "/" +
                            (card.snapshot.vertical_gain * 100).toFixed(0) + "%" : "—")
                    color: card.muted; font.pixelSize: 11
                }
                RowLayout {
                    spacing: 8
                    Text {
                        objectName: "armedLabel"
                        text: !card.online ? "离线" : card.snapshot.armed ? "已解锁" : "已上锁"
                        color: !card.online || card.snapshot.armed ? card.warning : "#8de9bb"
                        font { pixelSize: 12; bold: true }
                    }
                    Text {
                        objectName: "controlLabel"
                        text: !card.valid ? "桥接断开" : !card.snapshot.hid_fresh ? "手柄断开" : !card.snapshot.sending ? "监视" :
                              !card.snapshot.control_ready ? "等待归中" : "控制中"
                        color: card.valid && card.snapshot.control_ready && card.snapshot.sending ? "#8de9bb" : card.warning
                        font.pixelSize: 11
                    }
                    Text { text: card.valid ? card.snapshot.allow_arm ? "A允许" : "A锁止" : ""; color: card.muted; font.pixelSize: 10 }
                }
            }
        }
        Rectangle { Layout.fillWidth: true; height: 1; color: "#456078" }
        Text {
            text: card.number(card.snapshot.voltage_v, 1, "V", false) + " · " + card.number(card.snapshot.current_a, 2, "A", false)
                  + " · " + card.resourcesText()
            color: card.muted; font.pixelSize: 10
        }
        Text {
            Layout.fillWidth: true
            visible: !card.valid || !card.online || card.snapshot.warning !== ""
            text: !card.valid ? "数据中断 / 车辆不匹配"
                  : !card.online ? "飞控回传超时"
                  : card.snapshot.warning || ""
            color: card.warning; font.pixelSize: 11; wrapMode: Text.Wrap
            maximumLineCount: 2; elide: Text.ElideRight
        }
    }
}

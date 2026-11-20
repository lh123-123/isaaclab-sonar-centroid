import QtQuick
import QGroundControl

// Reuse the installed QGC resource unchanged, not a redrawn compass or the
// bridge's heading snapshot. This preserves QGC's native active-vehicle path.
Item {
    id: instrument
    implicitWidth: card.implicitWidth + gap + nativeLeftInset + nativeCompass.width
    implicitHeight: Math.max(card.implicitHeight, nativeCompass.height)
    width: implicitWidth
    height: implicitHeight

    property real extraInset: 0
    property real extraValuesWidth: 0
    property var vehicle: QGroundControl.multiVehicleManager.activeVehicle
    readonly property real gap: 12
    readonly property real nativeLeftInset: nativeCompass.item ? nativeCompass.item.extraInset : 0

    ROVCompetitionCard {
        id: card
        objectName: "competitionCard"
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        activeVehicleId: instrument.vehicle ? instrument.vehicle.id : 0
    }

    Loader {
        id: nativeCompass
        objectName: "nativeCompassLoader"
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        width: item ? item.implicitWidth : 0
        height: item ? item.implicitHeight : 0
        source: "qrc:/qml/QGroundControl/FlightMap/Widgets/IntegratedCompassAttitude.qml"
        onLoaded: item.vehicle = Qt.binding(function() { return instrument.vehicle })
    }

    // Fail visibly rather than substituting a fake compass or heading source.
    Text {
        anchors.left: parent.left
        anchors.bottom: card.top
        visible: nativeCompass.status === Loader.Error
        text: "QGC原生罗盘加载失败"
        color: "#ffcc66"
        font.pixelSize: 11
    }
}

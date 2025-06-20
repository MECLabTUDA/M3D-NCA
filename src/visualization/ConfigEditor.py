
import os
from typing import List, Any

from enum import Enum, auto

from dotenv import load_dotenv

from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *

class InputTypeRestriction(Enum):
    """
        Represents type of restriction on input data, will be mapped to a fitting validator
    """
    NoRestriction = auto()
    Integer = auto()
    Decimal = auto()
    Regex = auto()

class ConfigOptionType(Enum):
    """
        Available types for config options, will be mapped to arguments for the respective
        validator
    """
    Text = auto(),
    FullPath = auto(),
    RelPath = auto(),
    Int = auto(),
    Decimal = auto(),
    TupleInt = auto(),
    TupleDecimal = auto(),
    ListInt = auto(),
    ListDecimal = auto(),
    Bool = auto(),
    Custom = auto()

class ConfigInputRestriction:

    trestrict: InputTypeRestriction
    args: Any

    def __init__(self, trestrict: InputTypeRestriction, args: Any = None):
        """
            Creates new restriction to be used in ConfigOption class
        """
        self.trestrict = trestrict
        self.args = args

    def __iter__(self):
        return iter((self.trestrict, self.args))

class ConfigOption():

    name: str
    dtype: ConfigOptionType
    restriction: ConfigInputRestriction
    default_value: Any
    value: Any

    # some common restrictions
    full_path_regex = "^(/+[^/ ]*)+/?$"
    rel_path_regex = "^(/*[^/ ]*)+/?$"
    boolean_regex = "^(?:[Tt]rue|[Ff]alse)$"
    int_tuple_regex = "^(\( *\d+(, *\d+)* *\))?$"
    fp_tuple_regex = "^(\( *\d+(\.\d+)*( *, *\d+(\.\d+)*)* *\))?$"
    fp_list_regex = "^(\[ *\d+(\.\d+)*( *, *\d+(\.\d+)*)* *\])?$"
    int_tuple_restriction = ConfigInputRestriction(InputTypeRestriction.Regex, (int_tuple_regex, ))
    fp_tuple_restriction = ConfigInputRestriction(InputTypeRestriction.Regex, (fp_tuple_regex, ))
    fp_list_restritction = ConfigInputRestriction(InputTypeRestriction.Regex, (fp_list_regex, ))
    full_path_restriction = ConfigInputRestriction(InputTypeRestriction.Regex, (full_path_regex, ))
    rel_path_restriction = ConfigInputRestriction(InputTypeRestriction.Regex, (rel_path_regex, ))
    boolean_restriction = ConfigInputRestriction(InputTypeRestriction.Regex, (boolean_regex, ))
    simple_int_restriction = ConfigInputRestriction(InputTypeRestriction.Integer)
    simple_fp_restriction = ConfigInputRestriction(InputTypeRestriction.Decimal)
    no_restriction = ConfigInputRestriction(InputTypeRestriction.NoRestriction)

    # restriction lookup for config types 
    restriction_lookup = {
        ConfigOptionType.Text:          no_restriction,
        ConfigOptionType.FullPath:      full_path_restriction,
        ConfigOptionType.RelPath:       rel_path_restriction,
        ConfigOptionType.Int:           simple_int_restriction,
        ConfigOptionType.Decimal:       simple_fp_restriction,
        ConfigOptionType.Bool:          boolean_restriction,
        ConfigOptionType.TupleDecimal:  fp_tuple_restriction,
        ConfigOptionType.TupleInt:      int_tuple_restriction,
        ConfigOptionType.ListDecimal:   fp_list_restritction,
    }

    def __init__(self, name: str, dtype: ConfigOptionType, default_value: Any):
        """
            Creates new config option with automatic type restriction lookiup

            Args:
                name: name of config option
                dtype: type of config option, taken from available types in enum
                default_value: default value for config option
        """
        self.name = name
        self.dtype = dtype
        self.restriction = ConfigOption.restriction_lookup[self.dtype]
        self.default_value = default_value
        self.value = default_value

    def update(self, value: Any):
        """
            Updates value of config option

            Args:
                value: new value for config option
        """
        self.value = value

class ConfigEditor(QWidget):

    apply_btn: QPushButton

    validator_lookup = {
        InputTypeRestriction.Integer: QIntValidator,
        InputTypeRestriction.Decimal: QDoubleValidator,
        InputTypeRestriction.Regex: QRegularExpressionValidator
    }

    def __init__(self, options: List[ConfigOption], locale: QLocale.Language=QLocale.Language.English):
        """
            Creates a ConfigEditor from a list of ConfigOption[s]. The result is QWidget with vertically 
            arranged ConfigOption[s]. Each ConfigOption has an input field and a label for its name.
            Depending on the type of ConfigOption, there might be more GUI elements, e.g. Path types have 
            an additional browse button.

            Args:
                options: list of config options to create GUI from
                locale: needed for some validators, e.g. decimal points 
        """
        super().__init__()
        self.cfg = None
        self.validated = False
        self.options = options
        self.layout = QGridLayout()
        self.edits = []
        for i, option in enumerate(options):
            lbl: QLabel = QLabel(f"{option.name}: ")
            edit: QLineEdit = QLineEdit()
            option: ConfigOption
            edit.setText(str(option.default_value))
            edit.textChanged.connect(self.edit_changed)

            # add gui elements to layout
            self.layout.addWidget(lbl, i, 0)
            self.layout.addWidget(edit, i, 1)
            if option.dtype in (ConfigOptionType.FullPath, ConfigOptionType.RelPath):
                path_btn = QPushButton("...")
                path_btn.clicked.connect(lambda idx=i: self.path_btn(idx))
                self.layout.addWidget(path_btn, i, 2)
            
            self.edits.append(edit)
            # optional setup for input validation
            if option.restriction.trestrict != InputTypeRestriction.NoRestriction:
                if option.restriction.args is None:
                    validator = ConfigEditor.validator_lookup[option.restriction.trestrict]()
                    validator.setLocale(locale)
                else:
                    assert isinstance(option.restriction.args, tuple), "Arguments to validator have to be supplied in a tuple"
                    validator = ConfigEditor.validator_lookup[option.restriction.trestrict](*option.restriction.args)
                    validator.setLocale(locale)
                edit.setValidator(validator)

        # info label
        self.valid_lbl = QLabel(f"Validated:")
        self.validated_lbl = QLabel(f"{self.validated}")
        self.layout.addWidget(self.valid_lbl, len(self.options), 0)
        self.layout.addWidget(self.validated_lbl, len(self.options), 1, Qt.AlignmentFlag.AlignRight)
        # control buttons
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply)
        self.layout.addWidget(self.apply_btn, len(self.options) + 1, 0)
        self.close_btn = QPushButton("Close")
        self.layout.addWidget(self.close_btn, len(self.options) + 1, 2, 1, 1)

        self.setLayout(self.layout)

    def edit_changed(self):
        """
            Called whenever the input in one of the edit fields is changed.
        """
        self.validated = False
        self.validated_lbl.setText(f"{self.validated}")

    def path_btn(self, idx: int):
        """
            Called whenever a path browse button is clicked. Opens file dialog 
            and returns selected folder.
        """
        old_path = self.edits[idx].text()
        cwd = os.getcwd()
        dir = (cwd if old_path == "" else os.path.join(cwd, old_path))
        print(dir)
        new_path = QFileDialog.getExistingDirectory(self, caption="Select Folder", dir=dir)
        if not new_path == "":
            self.edits[idx].setText(new_path)

    def apply(self):
        """
            Tries to validate config, called when Apply button clicked.        
        """
        e : QLineEdit = None
        valid : bool = True
        for e in self.edits:
            if not e.hasAcceptableInput():
                valid = False
                print(f"Invalid: {e.text()}")
                break
        current_texts = [edit.text() for edit in self.edits]
        # construct config dictionary as strings
        self.cfg = dict()
        if valid:
            print("All config options were valid")
            for opt, val in zip(self.options, current_texts):
                if opt.dtype in (ConfigOptionType.Text, ConfigOptionType.FullPath, ConfigOptionType.RelPath):
                    self.cfg[opt.name] = val
                else:
                    self.cfg[opt.name] = eval(val)
            self.validated = True
        else:
            print("Config validation failed")

        self.validated_lbl.setText(f"{self.validated}")

    def get_config(self):
        if self.cfg == None or not self.validated:
            #TODO maybe show message box
            raise RuntimeError("Tried to return invalid config")
    
        return self.cfg

class ConfigEditorDialog(QDialog):

    def __init__(self, options: List[ConfigOption], parent=None):
        """Shows a custom dialog for editing the provided list of config options."""
        super().__init__()

        self.setWindowTitle("Config Editor")
        self.config_editor = ConfigEditor(options)

        self.layout = QVBoxLayout()

        self.layout.addWidget(self.config_editor)
        self.setLayout(self.layout)
        self.config_editor.close_btn.clicked.connect(self.accept)

    def accept(self) -> None:
        """If any config option not valid, will show a message box with that info."""
        if not self.config_editor.validated:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Cannot continue, config not validated")
            msg.setStandardButtons(QMessageBox.Ok)
            _ = msg.exec_()
            return
        else:
            return super().accept()

    @staticmethod
    def get_edited_config(options: List[ConfigOption], locale: QLocale.Language=QLocale.Language.English, parent=None) -> dict:
        """Shows this dialog and returns a dictionary created from the config options."""
        dialog = ConfigEditorDialog(options, locale)
        dialog.exec()
        return dialog.config_editor.get_config()
    


def config_options_from_dict(names_types: dict[str, ConfigOptionType], defaults: List[Any]) -> List[ConfigOption]:
    """
        Create a list of ConfigOption objects from names, types and default values.

        Args:
            names_types: dictionary mapping config option names to ConfigOptionType[s]
            defaults: list of default values, required to have same length as amount of names in
            the previous dictionary
    """
    opts = []
    for name, type, default in zip(names_types.keys(), names_types.values(), defaults):
        opts.append(ConfigOption(name=name, dtype=type, default_value=default))
    return opts

def default_config_options_to_dict(opts: List[ConfigOption]) -> dict:
    """
        Creates dictionary of config option names mapped to respective default values.

        Args:
            opts: list of ConfigOption[s]
    """
    cfg = {}
    for option in opts:
        cfg[option.name] = option.default_value
    return cfg

